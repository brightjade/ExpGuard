from src.classifier_models.aegis import AegisLlamaGuardPermissive, AegisLlamaGuardDefensive
from src.classifier_models.api_safety_classifiers import OpenAIModerationAPIClassifier
from src.classifier_models.base import SafetyClassifierBase, LegacySafetyClassifierBase, ConversationTurn, Role
from src.classifier_models.beaverdam import BeaverDam
from src.classifier_models.harmbench_classifier import HarmbenchClassifier, HarmbenchValidationClassifier
from src.classifier_models.llama_guard import LlamaGuardUserRequest, LlamaGuardModelResponse, LlamaGuard2, LlamaGuard3
from src.classifier_models.md_judge import MDJudgeResponseHarmClassifier
from src.classifier_models.wildguard import WildGuard
from src.classifier_models.perspective import PerspectiveAPIClassifier
from src.classifier_models.detoxify import DetoxifyClassifier
from src.classifier_models.azure import AzureContentSafetyClassifier
from src.classifier_models.shield_gemma import ShieldGemma
from src.classifier_models.expguard import ExpGuard


def load_classifier_model(model_name: str, **kwargs) -> SafetyClassifierBase:
    # LLM-based Classifiers
    if model_name == "ExpGuard":
        return ExpGuard(**kwargs)
    elif model_name == "LlamaGuardUserRequest":
        return LlamaGuardUserRequest(**kwargs)
    elif model_name == "LlamaGuardModelResponse":
        return LlamaGuardModelResponse(**kwargs)
    elif model_name == "LlamaGuard2":
        return LlamaGuard2(**kwargs)
    elif model_name == "LlamaGuard3":
        return LlamaGuard3(**kwargs)
    elif model_name == "AegisLlamaGuardPermissive":
        return AegisLlamaGuardPermissive(**kwargs)
    elif model_name == "AegisLlamaGuardDefensive":
        return AegisLlamaGuardDefensive(**kwargs)
    elif model_name == "ShieldGemma":
        return ShieldGemma(**kwargs)
    elif model_name == "HarmbenchClassifier":
        return HarmbenchClassifier(**kwargs)
    elif model_name == "HarmbenchValidationClassifier":
        return HarmbenchValidationClassifier(**kwargs)
    elif model_name == "MDJudgeResponseHarmClassifier":
        return MDJudgeResponseHarmClassifier(**kwargs)
    elif model_name == "BeaverDam":
        return BeaverDam(**kwargs)
    elif model_name == "WildGuard":
        return WildGuard(**kwargs)
    # API Classifiers
    elif model_name == "Detoxify":
        return DetoxifyClassifier()
    elif model_name == "PerspectiveAPI":
        return PerspectiveAPIClassifier()
    elif model_name == "OpenAIModeration":
        return OpenAIModerationAPIClassifier()
    elif model_name == "Azure":
        return AzureContentSafetyClassifier(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not found.")
