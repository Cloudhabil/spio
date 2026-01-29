"""
Mobile Configuration for Brahim Onion Agent APK

Supports:
- Android APK via Kivy/Buildozer
- Android APK via Chaquopy (Gradle plugin)
- iOS via BeeWare/Briefcase
- Cross-platform REST API server

Author: Elias Oulad Brahim
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DeploymentTarget(str, Enum):
    """Supported deployment targets."""
    ANDROID_APK = "android"
    IOS = "ios"
    WEB = "web"
    DESKTOP = "desktop"


class BuildTool(str, Enum):
    """Supported build tools."""
    BUILDOZER = "buildozer"      # Kivy -> APK
    CHAQUOPY = "chaquopy"        # Gradle Python plugin
    BRIEFCASE = "briefcase"      # BeeWare -> iOS/Android
    PYINSTALLER = "pyinstaller"  # Desktop executables


@dataclass
class MobileConfig:
    """Base mobile configuration."""
    app_name: str = "Brahim Onion Agent"
    package_name: str = "com.brahimlaws.agent"
    version: str = "1.3.0"

    # Display
    title: str = "BOA - Physics Calculator"
    icon: str = "assets/icon.png"
    presplash: str = "assets/splash.png"
    orientation: str = "portrait"

    # Permissions (Android)
    permissions: List[str] = field(default_factory=lambda: [
        "INTERNET",
        "ACCESS_NETWORK_STATE",
    ])

    # Features
    enable_onion_routing: bool = True
    enable_offline_mode: bool = True
    cache_calculations: bool = True

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8214  # Port = Sum constant (214) + padding

    # Model settings (for on-device inference)
    default_model: str = "gpt-4o-mini"
    fallback_to_local: bool = True


@dataclass
class APKConfig(MobileConfig):
    """Android APK specific configuration."""

    # Build settings
    build_tool: BuildTool = BuildTool.BUILDOZER
    target: DeploymentTarget = DeploymentTarget.ANDROID_APK

    # Android specific
    android_api: int = 33
    min_sdk: int = 24
    ndk_version: str = "25b"
    arch: List[str] = field(default_factory=lambda: ["arm64-v8a", "armeabi-v7a"])

    # Python requirements
    requirements: List[str] = field(default_factory=lambda: [
        "python3",
        "kivy",
        "requests",
        "pydantic",
    ])

    # Gradle (for Chaquopy)
    gradle_version: str = "8.2"
    chaquopy_version: str = "15.0.1"

    def to_buildozer_spec(self) -> str:
        """Generate buildozer.spec content."""
        return f"""[app]
title = {self.title}
package.name = {self.package_name.split('.')[-1]}
package.domain = {'.'.join(self.package_name.split('.')[:-1])}
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
version = {self.version}
requirements = {','.join(self.requirements)}
orientation = {self.orientation}
fullscreen = 0
android.permissions = {','.join(self.permissions)}
android.api = {self.android_api}
android.minapi = {self.min_sdk}
android.ndk = {self.ndk_version}
android.archs = {','.join(self.arch)}
android.allow_backup = True
ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.kivy_ios_branch = master
[buildozer]
log_level = 2
warn_on_root = 1
"""

    def to_chaquopy_gradle(self) -> str:
        """Generate build.gradle.kts content for Chaquopy."""
        return f"""plugins {{
    id("com.android.application")
    id("com.chaquo.python") version "{self.chaquopy_version}"
}}

android {{
    namespace = "{self.package_name}"
    compileSdk = {self.android_api}

    defaultConfig {{
        applicationId = "{self.package_name}"
        minSdk = {self.min_sdk}
        targetSdk = {self.android_api}
        versionCode = 1
        versionName = "{self.version}"

        ndk {{
            abiFilters += listOf({', '.join(f'"{a}"' for a in self.arch)})
        }}

        python {{
            version = "3.11"
            pip {{
                install("pydantic")
                install("requests")
            }}
        }}
    }}
}}

chaquopy {{
    defaultConfig {{
        python {{
            version = "3.11"
        }}
    }}
}}
"""


@dataclass
class OnionLayerConfig:
    """Configuration for onion routing layers."""

    # Layer C (Interface)
    layer_c_timeout: float = 5.0
    layer_c_max_input_length: int = 10000
    layer_c_intent_threshold: float = 0.7

    # Layer B (Symmetry)
    layer_b_strict_mode: bool = True
    layer_b_verify_on_every_call: bool = False
    layer_b_cache_verification: bool = True

    # Layer A (Core)
    layer_a_precision: int = 15  # decimal places
    layer_a_experimental_comparison: bool = True

    # Routing
    enable_parallel_layers: bool = False
    enable_layer_logging: bool = True
    trace_depth: int = 10


# =============================================================================
# INDUSTRY PRESETS
# =============================================================================

INDUSTRY_PRESETS: Dict[str, Dict[str, Any]] = {
    "particle_physics": {
        "name": "Particle Physics",
        "primary_tools": ["calculate_physics_constant", "calculate_yang_mills"],
        "default_calculations": ["fine_structure", "weinberg_angle"],
        "precision_requirement": "2 ppm",
        "organizations": ["CERN", "Fermilab", "DESY", "KEK"],
    },
    "cosmology": {
        "name": "Cosmology & Astrophysics",
        "primary_tools": ["calculate_cosmology"],
        "default_calculations": ["dark_matter", "dark_energy", "hubble"],
        "precision_requirement": "1%",
        "organizations": ["NASA", "ESA", "JWST", "Euclid"],
    },
    "quantum_computing": {
        "name": "Quantum Computing",
        "primary_tools": ["calculate_physics_constant", "apply_mirror_operator"],
        "default_calculations": ["fine_structure"],
        "precision_requirement": "sub-ppm",
        "organizations": ["IBM Quantum", "Google Quantum AI", "IonQ"],
    },
    "semiconductor": {
        "name": "Semiconductor / Metrology",
        "primary_tools": ["calculate_physics_constant"],
        "default_calculations": ["fine_structure"],
        "precision_requirement": "2 ppm",
        "organizations": ["NIST", "PTB", "Intel", "TSMC"],
    },
    "energy_research": {
        "name": "Energy Research",
        "primary_tools": ["calculate_yang_mills", "calculate_physics_constant"],
        "default_calculations": ["yang_mills", "proton_electron"],
        "precision_requirement": "1%",
        "organizations": ["ITER", "NIF", "Fusion startups"],
    },
    "ai_ml": {
        "name": "AI/ML Platforms",
        "primary_tools": ["get_brahim_sequence", "verify_axioms"],
        "default_calculations": ["sequence", "mirror_symmetry"],
        "precision_requirement": "exact",
        "organizations": ["OpenAI", "Anthropic", "DeepMind"],
    },
    "education": {
        "name": "Education & Academia",
        "primary_tools": ["*"],  # All tools
        "default_calculations": ["*"],  # All calculations
        "precision_requirement": "educational",
        "organizations": ["Universities", "Research Institutes"],
    },
}


def get_industry_config(industry: str) -> Dict[str, Any]:
    """Get configuration preset for an industry."""
    return INDUSTRY_PRESETS.get(industry.lower().replace(" ", "_"), INDUSTRY_PRESETS["education"])


# =============================================================================
# MANIFEST FOR APK
# =============================================================================

APK_MANIFEST = {
    "app": {
        "name": "Brahim Onion Agent",
        "version": "1.3.0",
        "description": "Multi-layer computational agent for physics calculations",
        "author": "Elias Oulad Brahim",
        "license": "TUL",
        "doi": "10.5281/zenodo.18356196",
    },
    "architecture": {
        "layers": [
            {"id": "C", "name": "Interface", "function": "Intent parsing, response formatting"},
            {"id": "B", "name": "Symmetry", "function": "Axiom verification, mirror operations"},
            {"id": "A", "name": "Core", "function": "Brahim mechanics calculations"},
        ],
        "routing": "onion",
        "flow": "User → C → A → B → C → User",
    },
    "capabilities": [
        {
            "name": "Fine Structure Constant",
            "function": "fine_structure_constant()",
            "output": "α⁻¹ = 137.036",
            "accuracy": "2 ppm",
            "industries": ["Particle Physics", "Metrology", "Semiconductor", "Quantum Computing"],
        },
        {
            "name": "Weinberg Angle",
            "function": "weinberg_angle()",
            "output": "sin²θ_W = 0.2308",
            "accuracy": "0.3%",
            "industries": ["Nuclear Physics", "Accelerator Labs", "Energy Research"],
        },
        {
            "name": "Muon-Electron Ratio",
            "function": "muon_electron_ratio()",
            "output": "m_μ/m_e = 206.8",
            "accuracy": "0.5%",
            "industries": ["Particle Physics", "Medical Imaging"],
        },
        {
            "name": "Proton-Electron Ratio",
            "function": "proton_electron_ratio()",
            "output": "m_p/m_e = 1836",
            "accuracy": "0.3%",
            "industries": ["Nuclear Physics", "Hydrogen Energy", "Fuel Cell R&D"],
        },
        {
            "name": "Cosmology Fractions",
            "function": "cosmic_fractions()",
            "output": "DM 27%, DE 68%, M 5%",
            "accuracy": "1%",
            "industries": ["Astrophysics", "Space Agencies", "Telescope Projects"],
        },
        {
            "name": "Yang-Mills Mass Gap",
            "function": "yang_mills_mass_gap()",
            "output": "Δ = 1721 MeV",
            "accuracy": "theoretical",
            "industries": ["QCD Research", "CERN", "Fermilab", "Theoretical Physics"],
        },
        {
            "name": "Mirror Operator",
            "function": "mirror_operator(x)",
            "output": "M(x) = 214 - x",
            "accuracy": "exact",
            "industries": ["Cryptography", "Data Symmetry", "Error Correction"],
        },
        {
            "name": "Sequence Generator",
            "function": "get_sequence()",
            "output": "[27, 42, 60, 75, 97, 117, 139, 154, 172, 187]",
            "accuracy": "exact",
            "industries": ["AI Training", "Pattern Recognition", "Financial Modeling"],
        },
        {
            "name": "Axiom Verification",
            "function": "verify_mirror_symmetry()",
            "output": "5/5 pairs verified (all sum to 214)",
            "accuracy": "exact",
            "industries": ["Quality Assurance", "Scientific Publishing", "Peer Review"],
        },
    ],
    "constants": {
        "brahim_sequence": [27, 42, 60, 75, 97, 117, 139, 154, 172, 187],
        "brahim_sequence_original": [27, 42, 60, 75, 97, 121, 136, 154, 172, 187],
        "sum": 214,
        "center": 107,
        "phi": 1.618033988749895,
        "delta_4": 0,
        "delta_5": 0,
        "asymmetry": 0,
        "regulator": 81,
    },
    "openai_compatible": True,
    "tools_count": 6,
    "handoff_agents": 4,
}
