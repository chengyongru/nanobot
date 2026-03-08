"""VirusTotal-based security scanning for skills."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
from loguru import logger

# VirusTotal API endpoint for file reports
VT_API_URL = "https://www.virustotal.com/api/v3/files/{hash}"


@dataclass
class ScanResult:
    """Result of a skill security scan."""

    safe: bool  # True if the skill is safe to load
    result: str  # "clean", "malicious", "unknown", "whitelisted", "cached"
    message: str  # Human-readable message


class SkillScanner:
    """
    Scans skill files using VirusTotal for malware detection.

    Scan flow:
    1. Check in-memory session cache (prevents redundant checks within same request)
    2. Check whitelist - if found, allow
    3. Check persistent cache - if found and valid, use cached result
    4. Query VirusTotal API
    5. Cache result and make decision
    """

    def __init__(
        self,
        vt_token: str,
        skill_security_config: dict,
        save_callback: Callable[[str, str, str, str], None] | None = None,
    ):
        """
        Initialize the skill scanner.

        Args:
            vt_token: VirusTotal API token.
            skill_security_config: Configuration dict with keys:
                - enabled: bool
                - unknown_ttl_seconds: int
                - whitelist: list[str]
                - scanned_hashes: dict
            save_callback: Optional callback to save scan results.
                Called as save_callback(sha256, result, scanned_at, skill_name).
        """
        self.vt_token = vt_token
        self.enabled = skill_security_config.get("enabled", True)
        self.unknown_ttl_seconds = skill_security_config.get("unknown_ttl_seconds", 86400)
        self.whitelist = set(skill_security_config.get("whitelist", []))
        self.scanned_hashes = skill_security_config.get("scanned_hashes", {})
        self.save_callback = save_callback
        # In-memory session cache to prevent redundant scans within same request
        self._session_cache: dict[str, ScanResult] = {}

    def check_skill(self, skill_path: Path) -> ScanResult:
        """
        Check if a skill file is safe to load.

        Args:
            skill_path: Path to the skill file (SKILL.md).

        Returns:
            ScanResult with safety decision and details.
        """
        skill_name = skill_path.parent.name

        # If security scanning is disabled, allow all
        if not self.enabled:
            return ScanResult(safe=True, result="disabled", message="Security scanning disabled")

        # Compute SHA256 hash
        try:
            content = skill_path.read_text(encoding="utf-8")
            sha256 = self._compute_sha256(content)
        except Exception as e:
            logger.warning(f"Failed to compute hash for skill '{skill_name}': {e}")
            return ScanResult(safe=True, result="error", message=f"Hash computation failed: {e}")

        # Check in-memory session cache first (prevents redundant scans within same request)
        if sha256 in self._session_cache:
            return self._session_cache[sha256]

        # Check whitelist
        if self._is_whitelisted(sha256):
            logger.debug(f"Skill '{skill_name}' in whitelist, skipping scan")
            result = ScanResult(safe=True, result="whitelisted", message="Skill is whitelisted")
            self._session_cache[sha256] = result
            return result

        # Check persistent cache
        cached = self._get_cached_result(sha256)
        if cached:
            cache_result, scanned_at = cached
            if cache_result == "malicious":
                logger.warning(f"Skill '{skill_name}' blocked: cached as malicious")
                result = ScanResult(safe=False, result="malicious", message="Cached as malicious")
                self._session_cache[sha256] = result
                return result
            elif cache_result == "clean":
                result = ScanResult(safe=True, result="clean", message="Cached as clean")
                self._session_cache[sha256] = result
                return result
            # For "unknown", continue to re-scan (TTL already handled in _get_cached_result)

        # Query VirusTotal
        logger.debug(f"Skill '{skill_name}' has hash {sha256[:16]}..., querying VT...")
        vt_result = self._query_virustotal(sha256)

        if vt_result is None:
            # API unavailable or error
            logger.warning(f"VirusTotal API unavailable, allowing skill '{skill_name}'")
            self._save_result(sha256, "unknown", skill_name)
            result = ScanResult(
                safe=True,
                result="unknown",
                message="VirusTotal API unavailable, allowed with caution"
            )
            self._session_cache[sha256] = result
            return result

        if vt_result == "not_found":
            # Hash not in VT database
            logger.info(f"Skill '{skill_name}' not in VT database, allowing with caution")
            self._save_result(sha256, "unknown", skill_name)
            result = ScanResult(
                safe=True,
                result="unknown",
                message="Hash not found in VirusTotal database"
            )
            self._session_cache[sha256] = result
            return result

        # We have a result from VT
        is_malicious, engine_count = vt_result
        self._save_result(sha256, "malicious" if is_malicious else "clean", skill_name)

        if is_malicious:
            logger.warning(
                f"Skill '{skill_name}' blocked: {engine_count[0]}/{engine_count[1]} "
                "engines detected malicious"
            )
            result = ScanResult(
                safe=False,
                result="malicious",
                message=f"{engine_count[0]}/{engine_count[1]} engines detected malicious"
            )
        else:
            logger.debug(f"Skill '{skill_name}' passed security scan")
            result = ScanResult(safe=True, result="clean", message="No malicious content detected")

        self._session_cache[sha256] = result
        return result

    def _compute_sha256(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _query_virustotal(self, sha256: str) -> tuple[bool, tuple[int, int]] | str | None:
        """
        Query VirusTotal API for a file hash.

        Args:
            sha256: SHA256 hash to query.

        Returns:
            - (is_malicious, (malicious_count, total_count)) if found
            - "not_found" if hash not in database
            - None if API unavailable or error
        """
        if not self.vt_token:
            logger.debug("No VirusTotal token configured")
            return None

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    VT_API_URL.format(hash=sha256),
                    headers={"x-apikey": self.vt_token}
                )

                if response.status_code == 404:
                    return "not_found"

                if response.status_code == 429:
                    logger.warning("VirusTotal API quota exhausted")
                    return None

                response.raise_for_status()

                data = response.json()
                attributes = data.get("data", {}).get("attributes", {})
                last_analysis_stats = attributes.get("last_analysis_stats", {})

                malicious = last_analysis_stats.get("malicious", 0)
                suspicious = last_analysis_stats.get("suspicious", 0)
                harmless = last_analysis_stats.get("harmless", 0)
                undetected = last_analysis_stats.get("undetected", 0)

                total = malicious + suspicious + harmless + undetected
                is_malicious = malicious > 0 or suspicious > 0

                return (is_malicious, (malicious + suspicious, total))

        except httpx.TimeoutException:
            logger.warning("VirusTotal API timeout")
            return None
        except httpx.HTTPStatusError as e:
            logger.warning(f"VirusTotal API error: {e}")
            return None
        except Exception as e:
            logger.warning(f"VirusTotal API request failed: {e}")
            return None

    def _is_whitelisted(self, sha256: str) -> bool:
        """Check if a hash is in the whitelist."""
        return sha256 in self.whitelist

    def _get_cached_result(self, sha256: str) -> tuple[str, str] | None:
        """
        Get cached scan result for a hash.

        Returns:
            (result, scanned_at) tuple or None if not cached or expired.
        """
        entry = self.scanned_hashes.get(sha256)
        if not entry:
            return None

        # Support both dict and Pydantic model (ScannedHashEntry)
        if hasattr(entry, "result"):
            result = entry.result
            scanned_at = entry.scanned_at
        else:
            result = entry.get("result", "")
            scanned_at = entry.get("scanned_at", "")

        # For "unknown" results, check TTL
        if result == "unknown" and scanned_at:
            try:
                scanned_time = datetime.fromisoformat(scanned_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                age_seconds = (now - scanned_time).total_seconds()

                if age_seconds > self.unknown_ttl_seconds:
                    logger.debug(
                        f"Cached 'unknown' result expired (age: {int(age_seconds)}s > "
                        f"TTL: {self.unknown_ttl_seconds}s), re-scanning"
                    )
                    return None
            except (ValueError, TypeError):
                pass  # Invalid timestamp, treat as not cached

        return (result, scanned_at)

    def _save_result(self, sha256: str, result: str, skill_name: str) -> None:
        """
        Save a scan result to cache.

        Args:
            sha256: The file hash.
            result: The scan result ("clean", "malicious", "unknown").
            skill_name: The name of the skill for easier identification.
        """
        scanned_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Update local cache
        # Support both dict and Pydantic model storage
        try:
            from nanobot.config.schema import ScannedHashEntry
            entry = ScannedHashEntry(
                result=result,
                scanned_at=scanned_at,
                skill_name=skill_name,
            )
        except ImportError:
            entry = {
                "result": result,
                "scanned_at": scanned_at,
                "skill_name": skill_name,
            }
        self.scanned_hashes[sha256] = entry

        # Call save callback if provided
        if self.save_callback:
            try:
                self.save_callback(sha256, result, scanned_at, skill_name)
            except Exception as e:
                logger.warning(f"Failed to save scan result: {e}")
