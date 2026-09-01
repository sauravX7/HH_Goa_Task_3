"""Configuration management using Pydantic Settings and YAML configuration."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MatchingConfig(BaseModel):
    similarity_threshold: float = 0.60
    distance_threshold: float = 0.60
    embedding_dimension: int = 128
    face_detector_backend: str = "face_recognition"


class BlockchainConfig(BaseModel):
    network: str = "hardhat"
    rpc_urls: Dict[str, str] = Field(
        default_factory=lambda: {
            "hardhat": "http://127.0.0.1:8545",
            "anvil": "http://127.0.0.1:8545",
            "polygon_amoy": "https://rpc-amoy.polygon.technology",
        }
    )
    chain_ids: Dict[str, int] = Field(
        default_factory=lambda: {
            "hardhat": 31337,
            "anvil": 31337,
            "polygon_amoy": 80002,
        }
    )
    default_account_private_key: str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    contract_address: Optional[str] = None
    gas_limit: int = 3000000


class SearchConfig(BaseModel):
    provider_priority: List[str] = Field(
        default_factory=lambda: [
            "serpapi_lens",
            "bing_visual",
            "playwright_lens",
            "mock",
        ]
    )
    max_candidates: int = 10
    timeout_seconds: int = 15
    candidate_fetch_timeout_seconds: int = 10
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )


class PathsConfig(BaseModel):
    artifacts_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    demo_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "demo")
    face_crop_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "face_crop.jpg")
    screenshot_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "search_result.png")
    metadata_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "metadata.json")
    canonical_post_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "canonical_post.json")
    sha256_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "sha256.txt")
    keccak256_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "keccak256.txt")
    tx_receipt_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "tx_receipt.json")
    verification_report_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "verification_report.json")
    pipeline_log_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "pipeline_log.json")
    contracts_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "contracts")


class DemoConfig(BaseModel):
    enabled: bool = False
    step_delay_seconds: float = 1.5
    save_step_screenshots: bool = True


class PrivacyConfig(BaseModel):
    user_supplied_images_only: bool = True
    public_sources_only: bool = True
    store_hashes_only_on_chain: bool = True


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "FaceProvenancePipeline"
    version: str = "1.0.0"

    # Sub-configurations
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    blockchain: BlockchainConfig = Field(default_factory=BlockchainConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    # Top-level environment variable overrides
    blockchain_network: Optional[str] = Field(default=None, alias="BLOCKCHAIN_NETWORK")
    blockchain_rpc_url: Optional[str] = Field(default=None, alias="BLOCKCHAIN_RPC_URL")
    contract_address: Optional[str] = Field(default=None, alias="CONTRACT_ADDRESS")
    private_key: Optional[str] = Field(default=None, alias="PRIVATE_KEY")

    polygon_amoy_rpc_url: Optional[str] = Field(default=None, alias="POLYGON_AMOY_RPC_URL")
    polygon_amoy_private_key: Optional[str] = Field(default=None, alias="POLYGON_AMOY_PRIVATE_KEY")

    serpapi_api_key: Optional[str] = Field(default=None, alias="SERPAPI_API_KEY")
    bing_visual_search_api_key: Optional[str] = Field(default=None, alias="BING_VISUAL_SEARCH_API_KEY")

    similarity_threshold_env: Optional[float] = Field(default=None, alias="SIMILARITY_THRESHOLD")
    distance_threshold_env: Optional[float] = Field(default=None, alias="DISTANCE_THRESHOLD")
    face_detector_backend_env: Optional[str] = Field(default=None, alias="FACE_DETECTOR_BACKEND")

    demo_mode_env: Optional[bool] = Field(default=None, alias="DEMO_MODE")
    demo_step_delay_env: Optional[float] = Field(default=None, alias="DEMO_STEP_DELAY")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def effective_network(self) -> str:
        return self.blockchain_network or self.blockchain.network

    @property
    def effective_rpc_url(self) -> str:
        if self.blockchain_rpc_url:
            return self.blockchain_rpc_url
        net = self.effective_network
        return self.blockchain.rpc_urls.get(net, "http://127.0.0.1:8545")

    @property
    def effective_chain_id(self) -> int:
        net = self.effective_network
        return self.blockchain.chain_ids.get(net, 31337)

    @property
    def effective_private_key(self) -> str:
        if self.private_key:
            return self.private_key
        if self.effective_network == "polygon_amoy" and self.polygon_amoy_private_key:
            return self.polygon_amoy_private_key
        return self.blockchain.default_account_private_key

    @property
    def effective_contract_address(self) -> Optional[str]:
        return self.contract_address or self.blockchain.contract_address

    @property
    def effective_similarity_threshold(self) -> float:
        if self.similarity_threshold_env is not None:
            return self.similarity_threshold_env
        return self.matching.similarity_threshold

    @property
    def effective_distance_threshold(self) -> float:
        if self.distance_threshold_env is not None:
            return self.distance_threshold_env
        return self.matching.distance_threshold

    @property
    def effective_face_detector_backend(self) -> str:
        if self.face_detector_backend_env is not None:
            return self.face_detector_backend_env
        return self.matching.face_detector_backend

    @property
    def is_demo_mode(self) -> bool:
        if self.demo_mode_env is not None:
            return self.demo_mode_env
        return self.demo.enabled

    @property
    def effective_demo_step_delay(self) -> float:
        if self.demo_step_delay_env is not None:
            return self.demo_step_delay_env
        return self.demo.step_delay_seconds

    def ensure_artifact_directories(self) -> None:
        """Ensure all required artifact output directories exist."""
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.paths.demo_dir.mkdir(parents=True, exist_ok=True)


def load_config(config_yaml_path: Optional[Path] = None) -> AppSettings:
    """Load configuration from config.yaml with environment variable overrides."""
    yaml_path = config_yaml_path or (PROJECT_ROOT / "config.yaml")
    yaml_data: Dict[str, Any] = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded
        except Exception:
            pass

    # Build nested models from YAML dict if present
    matching_cfg = MatchingConfig(**yaml_data.get("matching", {}))
    blockchain_cfg = BlockchainConfig(**yaml_data.get("blockchain", {}))
    search_cfg = SearchConfig(**yaml_data.get("search", {}))
    demo_cfg = DemoConfig(**yaml_data.get("demo", {}))
    privacy_cfg = PrivacyConfig(**yaml_data.get("privacy", {}))

    # Path overrides
    paths_data = yaml_data.get("paths", {})
    resolved_paths: Dict[str, Path] = {}
    for k, v in paths_data.items():
        if isinstance(v, str):
            p = Path(v)
            resolved_paths[k] = p if p.is_absolute() else (PROJECT_ROOT / p)
    paths_cfg = PathsConfig(**resolved_paths)

    settings = AppSettings(
        app_name=yaml_data.get("app_name", "FaceProvenancePipeline"),
        version=yaml_data.get("version", "1.0.0"),
        matching=matching_cfg,
        blockchain=blockchain_cfg,
        search=search_cfg,
        paths=paths_cfg,
        demo=demo_cfg,
        privacy=privacy_cfg,
    )
    return settings


# Global default configuration instance
config = load_config()
