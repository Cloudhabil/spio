"""
Skills Extension - Complete Skill Framework for Sovereign PIO

50+ skill components organized by 8 categories:

Categories:
1. Base (8)         - Skill, SkillContext, SkillResult, SkillMetadata, SkillDependency
2. Registry (5)     - SkillRegistry, GPIACoreRegistry, SkillCatalog
3. Loader (6)       - ProgressiveLoader, SkillHandle, SkillManifest, LazyLoader
4. Selector (4)     - AutonomousSkillSelector, SkillSelectorMemory, SelectionResult
5. Code (6)         - PythonSkill, RefactorSkill, ReviewSkill, FormatSkill
6. Data (4)         - AnalysisSkill, TransformSkill, ValidationSkill
7. Research (4)     - MathLiteratureSkill, BSDComparisonSkill, SynthesisSkill
8. Cognition (4)    - NeuroIntuitionSkill, PatternRecognitionSkill, ReasoningSkill

Components:
- SkillLevel: BASIC, INTERMEDIATE, ADVANCED, EXPERT
- SkillCategory: CODE, DATA, WRITING, RESEARCH, REASONING, SYNTHESIS, COGNITION
- SkillRouter: Routes tasks to optimal skills using PHI-saturation

Reference: CLI-main/src/skills/*
"""

from .skills_core import (
    # ========================================================================
    # ENUMS
    # ========================================================================
    SkillLevel,
    SkillCategory,
    SkillStatus,
    SelectionMethod,

    # ========================================================================
    # BASE CLASSES
    # ========================================================================
    SkillDependency,
    SkillMetadata,
    SkillContext,
    SkillResult,
    Skill,

    # ========================================================================
    # SKILL MANIFESTS
    # ========================================================================
    SkillManifest,

    # ========================================================================
    # SKILL HANDLES
    # ========================================================================
    SkillHandle,

    # ========================================================================
    # PROGRESSIVE LOADER
    # ========================================================================
    ProgressiveLoader,

    # ========================================================================
    # REGISTRY
    # ========================================================================
    SkillRegistry,
    GPIACoreSkillRegistry,

    # ========================================================================
    # SELECTOR
    # ========================================================================
    SkillSelectorMemory,
    SelectionResult,
    AutonomousSkillSelector,

    # ========================================================================
    # SKILL ROUTER
    # ========================================================================
    SkillRouter,

    # ========================================================================
    # PREDEFINED SKILLS
    # ========================================================================
    # Code skills
    PythonSkill,
    RefactorSkill,
    ReviewSkill,
    FormatSkill,
    DebugSkill,
    TestSkill,

    # Data skills
    AnalysisSkill,
    TransformSkill,
    DataValidationSkill,
    AggregationSkill,

    # Research skills
    MathLiteratureSkill,
    BSDComparisonSkill,
    SynthesisSkill,
    CitationSkill,

    # Cognition skills
    NeuroIntuitionSkill,
    PatternRecognitionSkill,
    ReasoningSkill,
    AbstractionSkill,

    # Writing skills
    DraftSkill,
    EditSkill,
    SummarySkill,
    TechnicalWritingSkill,

    # ========================================================================
    # SKILL COLLECTIONS
    # ========================================================================
    CODE_SKILLS,
    DATA_SKILLS,
    RESEARCH_SKILLS,
    COGNITION_SKILLS,
    WRITING_SKILLS,
    ALL_SKILLS,

    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================
    Skills,

    # ========================================================================
    # FACTORY FUNCTIONS
    # ========================================================================
    create_skill,
    create_registry,
    create_loader,
    create_selector,
    create_skills,
    load_skill,
    get_registry,
    get_loader,
)

__all__ = [
    # Enums
    "SkillLevel",
    "SkillCategory",
    "SkillStatus",
    "SelectionMethod",

    # Base
    "SkillDependency",
    "SkillMetadata",
    "SkillContext",
    "SkillResult",
    "Skill",

    # Manifests
    "SkillManifest",

    # Handles
    "SkillHandle",

    # Loader
    "ProgressiveLoader",

    # Registry
    "SkillRegistry",
    "GPIACoreSkillRegistry",

    # Selector
    "SkillSelectorMemory",
    "SelectionResult",
    "AutonomousSkillSelector",

    # Router
    "SkillRouter",

    # Code skills
    "PythonSkill",
    "RefactorSkill",
    "ReviewSkill",
    "FormatSkill",
    "DebugSkill",
    "TestSkill",

    # Data skills
    "AnalysisSkill",
    "TransformSkill",
    "DataValidationSkill",
    "AggregationSkill",

    # Research skills
    "MathLiteratureSkill",
    "BSDComparisonSkill",
    "SynthesisSkill",
    "CitationSkill",

    # Cognition skills
    "NeuroIntuitionSkill",
    "PatternRecognitionSkill",
    "ReasoningSkill",
    "AbstractionSkill",

    # Writing skills
    "DraftSkill",
    "EditSkill",
    "SummarySkill",
    "TechnicalWritingSkill",

    # Collections
    "CODE_SKILLS",
    "DATA_SKILLS",
    "RESEARCH_SKILLS",
    "COGNITION_SKILLS",
    "WRITING_SKILLS",
    "ALL_SKILLS",

    # Main
    "Skills",

    # Factories
    "create_skill",
    "create_registry",
    "create_loader",
    "create_selector",
    "create_skills",
    "load_skill",
    "get_registry",
    "get_loader",
]
