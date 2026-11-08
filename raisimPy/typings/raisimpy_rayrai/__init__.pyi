from collections.abc import Sequence
import datetime
import enum
import os
import raisimpy
from typing import Annotated, TypeAlias, overload

import numpy
from numpy.typing import NDArray

from . import assimp as assimp, color as color


class RayraiWindow:
    def __init__(self, world: raisimpy.World, width: int = 800, height: int = 600, visible: bool = True, title: str = 'RaiSim Rayrai', threadingMode: RayraiWindow.ThreadingMode = RayraiWindow.ThreadingMode.MultiThread, shaderBinaryCacheEnabled: bool = True, shaderBinaryCacheDirectory: str = '', shaderCompileThreadCount: int = 1, logShaderBinaryCacheOperations: bool = False) -> None:
        """
        Create an in-process renderer. visible=False creates a hidden context for image capture. Use and close on this thread.
        """

    class CaptureFilteredReflectionProbeOptions:
        def __init__(self) -> None: ...

        @property
        def position(self) -> numpy.ndarray: ...

        @position.setter
        def position(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def strength(self) -> float: ...

        @strength.setter
        def strength(self, arg: float, /) -> None: ...

        @property
        def capture(self) -> RayraiWindow.ReflectionProbeCaptureSettings | None: ...

        @capture.setter
        def capture(self, arg: RayraiWindow.ReflectionProbeCaptureSettings, /) -> None: ...

        @property
        def filter(self) -> RayraiWindow.ReflectionProbeFilterSettings | None: ...

        @filter.setter
        def filter(self, arg: RayraiWindow.ReflectionProbeFilterSettings, /) -> None: ...

    class ConservativeOcclusionCullingDiagnostics:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def used(self) -> bool: ...

        @used.setter
        def used(self, arg: bool, /) -> None: ...

        @property
        def supported(self) -> bool: ...

        @supported.setter
        def supported(self, arg: bool, /) -> None: ...

        @property
        def hizEnabled(self) -> bool: ...

        @hizEnabled.setter
        def hizEnabled(self, arg: bool, /) -> None: ...

        @property
        def hizUsed(self) -> bool: ...

        @hizUsed.setter
        def hizUsed(self, arg: bool, /) -> None: ...

        @property
        def gpuHizEnabled(self) -> bool: ...

        @gpuHizEnabled.setter
        def gpuHizEnabled(self, arg: bool, /) -> None: ...

        @property
        def gpuHizUsed(self) -> bool: ...

        @gpuHizUsed.setter
        def gpuHizUsed(self, arg: bool, /) -> None: ...

        @property
        def candidates(self) -> int: ...

        @candidates.setter
        def candidates(self, arg: int, /) -> None: ...

        @property
        def queries(self) -> int: ...

        @queries.setter
        def queries(self, arg: int, /) -> None: ...

        @property
        def occluded(self) -> int: ...

        @occluded.setter
        def occluded(self, arg: int, /) -> None: ...

        @property
        def drawn(self) -> int: ...

        @drawn.setter
        def drawn(self, arg: int, /) -> None: ...

        @property
        def hizTests(self) -> int: ...

        @hizTests.setter
        def hizTests(self, arg: int, /) -> None: ...

        @property
        def hizOccluded(self) -> int: ...

        @hizOccluded.setter
        def hizOccluded(self, arg: int, /) -> None: ...

        @property
        def hizDepthReadbacks(self) -> int: ...

        @hizDepthReadbacks.setter
        def hizDepthReadbacks(self, arg: int, /) -> None: ...

        @property
        def hizMipLevels(self) -> int: ...

        @hizMipLevels.setter
        def hizMipLevels(self, arg: int, /) -> None: ...

        @property
        def gpuHizDrawCalls(self) -> int: ...

        @gpuHizDrawCalls.setter
        def gpuHizDrawCalls(self, arg: int, /) -> None: ...

        @property
        def gpuHizDepthBuilds(self) -> int: ...

        @gpuHizDepthBuilds.setter
        def gpuHizDepthBuilds(self, arg: int, /) -> None: ...

        @property
        def gpuHizComputeDepthBuilds(self) -> int: ...

        @gpuHizComputeDepthBuilds.setter
        def gpuHizComputeDepthBuilds(self, arg: int, /) -> None: ...

        @property
        def gpuHizRasterDepthBuilds(self) -> int: ...

        @gpuHizRasterDepthBuilds.setter
        def gpuHizRasterDepthBuilds(self, arg: int, /) -> None: ...

        @property
        def gpuHizMipLevels(self) -> int: ...

        @gpuHizMipLevels.setter
        def gpuHizMipLevels(self, arg: int, /) -> None: ...

        @property
        def fallbackCode(self) -> int: ...

        @fallbackCode.setter
        def fallbackCode(self, arg: int, /) -> None: ...

        @property
        def hizFallbackCode(self) -> int: ...

        @hizFallbackCode.setter
        def hizFallbackCode(self, arg: int, /) -> None: ...

        @property
        def gpuHizFallbackCode(self) -> int: ...

        @gpuHizFallbackCode.setter
        def gpuHizFallbackCode(self, arg: int, /) -> None: ...

    class DebugPassCapture:
        def __init__(self) -> None: ...

        @property
        def fileName(self) -> str: ...

        @fileName.setter
        def fileName(self, arg: str, /) -> None: ...

        @property
        def label(self) -> str: ...

        @label.setter
        def label(self, arg: str, /) -> None: ...

        @property
        def pbrDebugOutput(self) -> int: ...

        @pbrDebugOutput.setter
        def pbrDebugOutput(self, arg: int, /) -> None: ...

        @property
        def postProcessDebugMode(self) -> PostProcessDebugMode: ...

        @postProcessDebugMode.setter
        def postProcessDebugMode(self, arg: PostProcessDebugMode, /) -> None: ...

        @property
        def width(self) -> int: ...

        @width.setter
        def width(self, arg: int, /) -> None: ...

        @property
        def height(self) -> int: ...

        @height.setter
        def height(self, arg: int, /) -> None: ...

        @property
        def rgba(self) -> list[int]: ...

        @rgba.setter
        def rgba(self, arg: Sequence[int], /) -> None: ...

    class DirectionalShadowCascade:
        def __init__(self) -> None: ...

        @property
        def index(self) -> int: ...

        @index.setter
        def index(self, arg: int, /) -> None: ...

        @property
        def nearPlane(self) -> float: ...

        @nearPlane.setter
        def nearPlane(self, arg: float, /) -> None: ...

        @property
        def farPlane(self) -> float: ...

        @farPlane.setter
        def farPlane(self, arg: float, /) -> None: ...

        @property
        def splitDepth(self) -> float: ...

        @splitDepth.setter
        def splitDepth(self, arg: float, /) -> None: ...

        @property
        def texelWorldSize(self) -> float: ...

        @texelWorldSize.setter
        def texelWorldSize(self, arg: float, /) -> None: ...

        @property
        def center(self) -> numpy.ndarray: ...

        @center.setter
        def center(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lightSpaceMin(self) -> numpy.ndarray: ...

        @lightSpaceMin.setter
        def lightSpaceMin(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lightSpaceMax(self) -> numpy.ndarray: ...

        @lightSpaceMax.setter
        def lightSpaceMax(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lightSpaceHalfExtents(self) -> numpy.ndarray: ...

        @lightSpaceHalfExtents.setter
        def lightSpaceHalfExtents(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lightSpaceMatrix(self) -> numpy.ndarray: ...

        @lightSpaceMatrix.setter
        def lightSpaceMatrix(self, arg: numpy.ndarray, /) -> None: ...

    class DirectionalShadowCascadePlan:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def requestedCascadeCount(self) -> int: ...

        @requestedCascadeCount.setter
        def requestedCascadeCount(self, arg: int, /) -> None: ...

        @property
        def cascadeCount(self) -> int: ...

        @cascadeCount.setter
        def cascadeCount(self, arg: int, /) -> None: ...

        @property
        def nearPlane(self) -> float: ...

        @nearPlane.setter
        def nearPlane(self, arg: float, /) -> None: ...

        @property
        def farPlane(self) -> float: ...

        @farPlane.setter
        def farPlane(self, arg: float, /) -> None: ...

        @property
        def maxDistance(self) -> float: ...

        @maxDistance.setter
        def maxDistance(self, arg: float, /) -> None: ...

        @property
        def shadowResolution(self) -> float: ...

        @shadowResolution.setter
        def shadowResolution(self, arg: float, /) -> None: ...

        @property
        def stableTexelSnap(self) -> bool: ...

        @stableTexelSnap.setter
        def stableTexelSnap(self, arg: bool, /) -> None: ...

        @property
        def splitOverridesEnabled(self) -> bool: ...

        @splitOverridesEnabled.setter
        def splitOverridesEnabled(self, arg: bool, /) -> None: ...

        @property
        def splitOverrides(self) -> list[float]: ...

        @splitOverrides.setter
        def splitOverrides(self, arg: Sequence[float], /) -> None: ...

        @property
        def cascades(self) -> list[RayraiWindow.DirectionalShadowCascade]: ...

        @cascades.setter
        def cascades(self, arg: Sequence[RayraiWindow.DirectionalShadowCascade], /) -> None: ...

    class DynamicQualityRecommendation:
        def __init__(self) -> None: ...

        @property
        def changed(self) -> bool: ...

        @changed.setter
        def changed(self, arg: bool, /) -> None: ...

        @property
        def fillRateBound(self) -> bool: ...

        @fillRateBound.setter
        def fillRateBound(self, arg: bool, /) -> None: ...

        @property
        def shadowBound(self) -> bool: ...

        @shadowBound.setter
        def shadowBound(self, arg: bool, /) -> None: ...

        @property
        def postprocessBound(self) -> bool: ...

        @postprocessBound.setter
        def postprocessBound(self, arg: bool, /) -> None: ...

        @property
        def totalGpuMs(self) -> float: ...

        @totalGpuMs.setter
        def totalGpuMs(self, arg: float, /) -> None: ...

        @property
        def shadowGpuMs(self) -> float: ...

        @shadowGpuMs.setter
        def shadowGpuMs(self, arg: float, /) -> None: ...

        @property
        def sceneGpuMs(self) -> float: ...

        @sceneGpuMs.setter
        def sceneGpuMs(self, arg: float, /) -> None: ...

        @property
        def postprocessGpuMs(self) -> float: ...

        @postprocessGpuMs.setter
        def postprocessGpuMs(self, arg: float, /) -> None: ...

        @property
        def targetFrameMs(self) -> float: ...

        @targetFrameMs.setter
        def targetFrameMs(self, arg: float, /) -> None: ...

        @property
        def recommendedRenderScale(self) -> float: ...

        @recommendedRenderScale.setter
        def recommendedRenderScale(self, arg: float, /) -> None: ...

        @property
        def recommendedUpdateShadowsEveryFrame(self) -> bool: ...

        @recommendedUpdateShadowsEveryFrame.setter
        def recommendedUpdateShadowsEveryFrame(self, arg: bool, /) -> None: ...

        @property
        def recommendedBloomQuality(self) -> int: ...

        @recommendedBloomQuality.setter
        def recommendedBloomQuality(self, arg: int, /) -> None: ...

        @property
        def recommendedScreenSpaceAoSamples(self) -> int: ...

        @recommendedScreenSpaceAoSamples.setter
        def recommendedScreenSpaceAoSamples(self, arg: int, /) -> None: ...

        @property
        def recommendedScreenSpaceAoDenoiseEnabled(self) -> bool: ...

        @recommendedScreenSpaceAoDenoiseEnabled.setter
        def recommendedScreenSpaceAoDenoiseEnabled(self, arg: bool, /) -> None: ...

        @property
        def recommendedSettings(self) -> RenderQualitySettings: ...

        @recommendedSettings.setter
        def recommendedSettings(self, arg: RenderQualitySettings, /) -> None: ...

        @property
        def reason(self) -> str: ...

        @reason.setter
        def reason(self, arg: str, /) -> None: ...

    class EnvironmentBackgroundDiagnostics:
        def __init__(self) -> None: ...

        @property
        def drawn(self) -> bool: ...

        @drawn.setter
        def drawn(self, arg: bool, /) -> None: ...

        @property
        def deferred(self) -> bool: ...

        @deferred.setter
        def deferred(self, arg: bool, /) -> None: ...

        @property
        def depthTested(self) -> bool: ...

        @depthTested.setter
        def depthTested(self, arg: bool, /) -> None: ...

    class EnvironmentSidecarSettings:
        def __init__(self) -> None: ...

        @property
        def hasEnvironmentIntensity(self) -> bool: ...

        @hasEnvironmentIntensity.setter
        def hasEnvironmentIntensity(self, arg: bool, /) -> None: ...

        @property
        def environmentIntensity(self) -> float: ...

        @environmentIntensity.setter
        def environmentIntensity(self, arg: float, /) -> None: ...

        @property
        def hasEnvironmentRotation(self) -> bool: ...

        @hasEnvironmentRotation.setter
        def hasEnvironmentRotation(self, arg: bool, /) -> None: ...

        @property
        def environmentRotation(self) -> float: ...

        @environmentRotation.setter
        def environmentRotation(self, arg: float, /) -> None: ...

        @property
        def hasSkyTint(self) -> bool: ...

        @hasSkyTint.setter
        def hasSkyTint(self, arg: bool, /) -> None: ...

        @property
        def skyTint(self) -> numpy.ndarray: ...

        @skyTint.setter
        def skyTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def hasHorizonTint(self) -> bool: ...

        @hasHorizonTint.setter
        def hasHorizonTint(self, arg: bool, /) -> None: ...

        @property
        def horizonTint(self) -> numpy.ndarray: ...

        @horizonTint.setter
        def horizonTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def hasGroundTint(self) -> bool: ...

        @hasGroundTint.setter
        def hasGroundTint(self, arg: bool, /) -> None: ...

        @property
        def groundTint(self) -> numpy.ndarray: ...

        @groundTint.setter
        def groundTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def hasHorizonStrength(self) -> bool: ...

        @hasHorizonStrength.setter
        def hasHorizonStrength(self, arg: bool, /) -> None: ...

        @property
        def horizonStrength(self) -> float: ...

        @horizonStrength.setter
        def horizonStrength(self, arg: float, /) -> None: ...

        @property
        def hasBackgroundTint(self) -> bool: ...

        @hasBackgroundTint.setter
        def hasBackgroundTint(self, arg: bool, /) -> None: ...

        @property
        def backgroundTint(self) -> numpy.ndarray: ...

        @backgroundTint.setter
        def backgroundTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def hasBackgroundExposureScale(self) -> bool: ...

        @hasBackgroundExposureScale.setter
        def hasBackgroundExposureScale(self, arg: bool, /) -> None: ...

        @property
        def backgroundExposureScale(self) -> float: ...

        @backgroundExposureScale.setter
        def backgroundExposureScale(self, arg: float, /) -> None: ...

        @property
        def hasBackgroundMapPath(self) -> bool: ...

        @hasBackgroundMapPath.setter
        def hasBackgroundMapPath(self, arg: bool, /) -> None: ...

        @property
        def backgroundMapPath(self) -> str: ...

        @backgroundMapPath.setter
        def backgroundMapPath(self, arg: str, /) -> None: ...

        @property
        def hasBackgroundMapExposure(self) -> bool: ...

        @hasBackgroundMapExposure.setter
        def hasBackgroundMapExposure(self, arg: bool, /) -> None: ...

        @property
        def backgroundMapExposure(self) -> float: ...

        @backgroundMapExposure.setter
        def backgroundMapExposure(self, arg: float, /) -> None: ...

        @property
        def hasProceduralSkyEnabled(self) -> bool: ...

        @hasProceduralSkyEnabled.setter
        def hasProceduralSkyEnabled(self, arg: bool, /) -> None: ...

        @property
        def proceduralSkyEnabled(self) -> bool: ...

        @proceduralSkyEnabled.setter
        def proceduralSkyEnabled(self, arg: bool, /) -> None: ...

        @property
        def hasProceduralSkySunStrength(self) -> bool: ...

        @hasProceduralSkySunStrength.setter
        def hasProceduralSkySunStrength(self, arg: bool, /) -> None: ...

        @property
        def proceduralSkySunStrength(self) -> float: ...

        @proceduralSkySunStrength.setter
        def proceduralSkySunStrength(self, arg: float, /) -> None: ...

        @property
        def hasProceduralSkySunSize(self) -> bool: ...

        @hasProceduralSkySunSize.setter
        def hasProceduralSkySunSize(self, arg: bool, /) -> None: ...

        @property
        def proceduralSkySunSize(self) -> float: ...

        @proceduralSkySunSize.setter
        def proceduralSkySunSize(self, arg: float, /) -> None: ...

        @property
        def hasSunDirection(self) -> bool: ...

        @hasSunDirection.setter
        def hasSunDirection(self, arg: bool, /) -> None: ...

        @property
        def sunDirection(self) -> numpy.ndarray: ...

        @sunDirection.setter
        def sunDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def hasSunColor(self) -> bool: ...

        @hasSunColor.setter
        def hasSunColor(self, arg: bool, /) -> None: ...

        @property
        def sunColor(self) -> numpy.ndarray: ...

        @sunColor.setter
        def sunColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def hasSunIntensity(self) -> bool: ...

        @hasSunIntensity.setter
        def hasSunIntensity(self, arg: bool, /) -> None: ...

        @property
        def sunIntensity(self) -> float: ...

        @sunIntensity.setter
        def sunIntensity(self, arg: float, /) -> None: ...

        @property
        def hasExposure(self) -> bool: ...

        @hasExposure.setter
        def hasExposure(self, arg: bool, /) -> None: ...

        @property
        def exposure(self) -> float: ...

        @exposure.setter
        def exposure(self, arg: float, /) -> None: ...

        @property
        def hasBloomEnabled(self) -> bool: ...

        @hasBloomEnabled.setter
        def hasBloomEnabled(self, arg: bool, /) -> None: ...

        @property
        def bloomEnabled(self) -> bool: ...

        @bloomEnabled.setter
        def bloomEnabled(self, arg: bool, /) -> None: ...

        @property
        def hasBloomThreshold(self) -> bool: ...

        @hasBloomThreshold.setter
        def hasBloomThreshold(self, arg: bool, /) -> None: ...

        @property
        def bloomThreshold(self) -> float: ...

        @bloomThreshold.setter
        def bloomThreshold(self, arg: float, /) -> None: ...

        @property
        def hasBloomStrength(self) -> bool: ...

        @hasBloomStrength.setter
        def hasBloomStrength(self, arg: bool, /) -> None: ...

        @property
        def bloomStrength(self) -> float: ...

        @bloomStrength.setter
        def bloomStrength(self, arg: float, /) -> None: ...

        @property
        def hasBloomRadius(self) -> bool: ...

        @hasBloomRadius.setter
        def hasBloomRadius(self, arg: bool, /) -> None: ...

        @property
        def bloomRadius(self) -> float: ...

        @bloomRadius.setter
        def bloomRadius(self, arg: float, /) -> None: ...

        @property
        def hasBloomKnee(self) -> bool: ...

        @hasBloomKnee.setter
        def hasBloomKnee(self, arg: bool, /) -> None: ...

        @property
        def bloomKnee(self) -> float: ...

        @bloomKnee.setter
        def bloomKnee(self, arg: float, /) -> None: ...

        @property
        def hasBloomQuality(self) -> bool: ...

        @hasBloomQuality.setter
        def hasBloomQuality(self, arg: bool, /) -> None: ...

        @property
        def bloomQuality(self) -> int: ...

        @bloomQuality.setter
        def bloomQuality(self, arg: int, /) -> None: ...

        @property
        def hasBloomSourceClamp(self) -> bool: ...

        @hasBloomSourceClamp.setter
        def hasBloomSourceClamp(self, arg: bool, /) -> None: ...

        @property
        def bloomSourceClamp(self) -> float: ...

        @bloomSourceClamp.setter
        def bloomSourceClamp(self, arg: float, /) -> None: ...

        @property
        def hasBloomAnamorphic(self) -> bool: ...

        @hasBloomAnamorphic.setter
        def hasBloomAnamorphic(self, arg: bool, /) -> None: ...

        @property
        def bloomAnamorphic(self) -> float: ...

        @bloomAnamorphic.setter
        def bloomAnamorphic(self, arg: float, /) -> None: ...

        @property
        def hasBloomDirtStrength(self) -> bool: ...

        @hasBloomDirtStrength.setter
        def hasBloomDirtStrength(self, arg: bool, /) -> None: ...

        @property
        def bloomDirtStrength(self) -> float: ...

        @bloomDirtStrength.setter
        def bloomDirtStrength(self, arg: float, /) -> None: ...

        @property
        def hasBloomDirtScale(self) -> bool: ...

        @hasBloomDirtScale.setter
        def hasBloomDirtScale(self, arg: bool, /) -> None: ...

        @property
        def bloomDirtScale(self) -> float: ...

        @bloomDirtScale.setter
        def bloomDirtScale(self, arg: float, /) -> None: ...

        @property
        def hasBloomBlendMode(self) -> bool: ...

        @hasBloomBlendMode.setter
        def hasBloomBlendMode(self, arg: bool, /) -> None: ...

        @property
        def bloomBlendMode(self) -> GlowBlendMode: ...

        @bloomBlendMode.setter
        def bloomBlendMode(self, arg: GlowBlendMode, /) -> None: ...

        @property
        def hasBloomMix(self) -> bool: ...

        @hasBloomMix.setter
        def hasBloomMix(self, arg: bool, /) -> None: ...

        @property
        def bloomMix(self) -> float: ...

        @bloomMix.setter
        def bloomMix(self, arg: float, /) -> None: ...

        @property
        def hasFogDensity(self) -> bool: ...

        @hasFogDensity.setter
        def hasFogDensity(self, arg: bool, /) -> None: ...

        @property
        def fogDensity(self) -> float: ...

        @fogDensity.setter
        def fogDensity(self, arg: float, /) -> None: ...

        @property
        def hasHeightFogEnabled(self) -> bool: ...

        @hasHeightFogEnabled.setter
        def hasHeightFogEnabled(self, arg: bool, /) -> None: ...

        @property
        def heightFogEnabled(self) -> bool: ...

        @heightFogEnabled.setter
        def heightFogEnabled(self, arg: bool, /) -> None: ...

        @property
        def hasHeightFogDensity(self) -> bool: ...

        @hasHeightFogDensity.setter
        def hasHeightFogDensity(self, arg: bool, /) -> None: ...

        @property
        def heightFogDensity(self) -> float: ...

        @heightFogDensity.setter
        def heightFogDensity(self, arg: float, /) -> None: ...

        @property
        def hasHeightFogBaseHeight(self) -> bool: ...

        @hasHeightFogBaseHeight.setter
        def hasHeightFogBaseHeight(self, arg: bool, /) -> None: ...

        @property
        def heightFogBaseHeight(self) -> float: ...

        @heightFogBaseHeight.setter
        def heightFogBaseHeight(self, arg: float, /) -> None: ...

        @property
        def hasHeightFogFalloff(self) -> bool: ...

        @hasHeightFogFalloff.setter
        def hasHeightFogFalloff(self, arg: bool, /) -> None: ...

        @property
        def heightFogFalloff(self) -> float: ...

        @heightFogFalloff.setter
        def heightFogFalloff(self, arg: float, /) -> None: ...

    class ExposureRecommendation:
        def __init__(self) -> None: ...

        @property
        def valid(self) -> bool: ...

        @valid.setter
        def valid(self, arg: bool, /) -> None: ...

        @property
        def changed(self) -> bool: ...

        @changed.setter
        def changed(self, arg: bool, /) -> None: ...

        @property
        def underexposed(self) -> bool: ...

        @underexposed.setter
        def underexposed(self, arg: bool, /) -> None: ...

        @property
        def overexposed(self) -> bool: ...

        @overexposed.setter
        def overexposed(self, arg: bool, /) -> None: ...

        @property
        def clippingRisk(self) -> bool: ...

        @clippingRisk.setter
        def clippingRisk(self, arg: bool, /) -> None: ...

        @property
        def currentExposure(self) -> float: ...

        @currentExposure.setter
        def currentExposure(self, arg: float, /) -> None: ...

        @property
        def recommendedExposure(self) -> float: ...

        @recommendedExposure.setter
        def recommendedExposure(self, arg: float, /) -> None: ...

        @property
        def recommendedExposureScale(self) -> float: ...

        @recommendedExposureScale.setter
        def recommendedExposureScale(self, arg: float, /) -> None: ...

        @property
        def targetMedianLuminance(self) -> float: ...

        @targetMedianLuminance.setter
        def targetMedianLuminance(self, arg: float, /) -> None: ...

        @property
        def targetP95Luminance(self) -> float: ...

        @targetP95Luminance.setter
        def targetP95Luminance(self, arg: float, /) -> None: ...

        @property
        def minExposure(self) -> float: ...

        @minExposure.setter
        def minExposure(self, arg: float, /) -> None: ...

        @property
        def maxExposure(self) -> float: ...

        @maxExposure.setter
        def maxExposure(self, arg: float, /) -> None: ...

        @property
        def reason(self) -> str: ...

        @reason.setter
        def reason(self, arg: str, /) -> None: ...

    class FoliageRenderDiagnostics:
        def __init__(self) -> None: ...

        @property
        def available(self) -> bool: ...

        @available.setter
        def available(self, arg: bool, /) -> None: ...

        @property
        def timingCaptureAvailable(self) -> bool: ...

        @timingCaptureAvailable.setter
        def timingCaptureAvailable(self, arg: bool, /) -> None: ...

        @property
        def sceneFoliageDrawCalls(self) -> int: ...

        @sceneFoliageDrawCalls.setter
        def sceneFoliageDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneFoliageOccludedDrawCalls(self) -> int: ...

        @sceneFoliageOccludedDrawCalls.setter
        def sceneFoliageOccludedDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneFoliageAuthoredInstanceVisits(self) -> int: ...

        @sceneFoliageAuthoredInstanceVisits.setter
        def sceneFoliageAuthoredInstanceVisits(self, arg: int, /) -> None: ...

        @property
        def sceneFoliageCandidateInstanceVisits(self) -> int: ...

        @sceneFoliageCandidateInstanceVisits.setter
        def sceneFoliageCandidateInstanceVisits(self, arg: int, /) -> None: ...

        @property
        def sceneFoliageRenderedInstanceVisits(self) -> int: ...

        @sceneFoliageRenderedInstanceVisits.setter
        def sceneFoliageRenderedInstanceVisits(self, arg: int, /) -> None: ...

        @property
        def shadowFoliageDrawCalls(self) -> int: ...

        @shadowFoliageDrawCalls.setter
        def shadowFoliageDrawCalls(self, arg: int, /) -> None: ...

        @property
        def shadowFoliageAuthoredInstanceVisits(self) -> int: ...

        @shadowFoliageAuthoredInstanceVisits.setter
        def shadowFoliageAuthoredInstanceVisits(self, arg: int, /) -> None: ...

        @property
        def shadowFoliageCandidateInstanceVisits(self) -> int: ...

        @shadowFoliageCandidateInstanceVisits.setter
        def shadowFoliageCandidateInstanceVisits(self, arg: int, /) -> None: ...

        @property
        def shadowFoliageRenderedInstanceVisits(self) -> int: ...

        @shadowFoliageRenderedInstanceVisits.setter
        def shadowFoliageRenderedInstanceVisits(self, arg: int, /) -> None: ...

        @property
        def sceneGpuCulledDrawCalls(self) -> int: ...

        @sceneGpuCulledDrawCalls.setter
        def sceneGpuCulledDrawCalls(self, arg: int, /) -> None: ...

        @property
        def shadowGpuCulledDrawCalls(self) -> int: ...

        @shadowGpuCulledDrawCalls.setter
        def shadowGpuCulledDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneGpuCullSkippedDrawCalls(self) -> int: ...

        @sceneGpuCullSkippedDrawCalls.setter
        def sceneGpuCullSkippedDrawCalls(self, arg: int, /) -> None: ...

        @property
        def shadowGpuCullSkippedDrawCalls(self) -> int: ...

        @shadowGpuCullSkippedDrawCalls.setter
        def shadowGpuCullSkippedDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneVisibleInstanceFrameCacheHits(self) -> int: ...

        @sceneVisibleInstanceFrameCacheHits.setter
        def sceneVisibleInstanceFrameCacheHits(self, arg: int, /) -> None: ...

        @property
        def sceneVisibleInstanceFrameCacheDispatches(self) -> int: ...

        @sceneVisibleInstanceFrameCacheDispatches.setter
        def sceneVisibleInstanceFrameCacheDispatches(self, arg: int, /) -> None: ...

        @property
        def sceneVisibleInstanceFrameCacheFallbackDrawCalls(self) -> int: ...

        @sceneVisibleInstanceFrameCacheFallbackDrawCalls.setter
        def sceneVisibleInstanceFrameCacheFallbackDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneClusteredDrawCalls(self) -> int: ...

        @sceneClusteredDrawCalls.setter
        def sceneClusteredDrawCalls(self, arg: int, /) -> None: ...

        @property
        def shadowClusteredDrawCalls(self) -> int: ...

        @shadowClusteredDrawCalls.setter
        def shadowClusteredDrawCalls(self, arg: int, /) -> None: ...

        @property
        def impostorDrawCalls(self) -> int: ...

        @impostorDrawCalls.setter
        def impostorDrawCalls(self, arg: int, /) -> None: ...

        @property
        def shadowLodDrawCalls(self) -> int: ...

        @shadowLodDrawCalls.setter
        def shadowLodDrawCalls(self, arg: int, /) -> None: ...

        @property
        def persistentSourceBufferDrawCalls(self) -> int: ...

        @persistentSourceBufferDrawCalls.setter
        def persistentSourceBufferDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneMultiDrawIndirectDrawCalls(self) -> int: ...

        @sceneMultiDrawIndirectDrawCalls.setter
        def sceneMultiDrawIndirectDrawCalls(self, arg: int, /) -> None: ...

        @property
        def shadowMultiDrawIndirectDrawCalls(self) -> int: ...

        @shadowMultiDrawIndirectDrawCalls.setter
        def shadowMultiDrawIndirectDrawCalls(self, arg: int, /) -> None: ...

        @property
        def multiDrawIndirectCommandCount(self) -> int: ...

        @multiDrawIndirectCommandCount.setter
        def multiDrawIndirectCommandCount(self, arg: int, /) -> None: ...

        @property
        def multiDrawIndirectFallbackDrawCalls(self) -> int: ...

        @multiDrawIndirectFallbackDrawCalls.setter
        def multiDrawIndirectFallbackDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneAlphaDepthPrepassDrawCalls(self) -> int: ...

        @sceneAlphaDepthPrepassDrawCalls.setter
        def sceneAlphaDepthPrepassDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneAlphaDepthPrepassInstanceVisits(self) -> int: ...

        @sceneAlphaDepthPrepassInstanceVisits.setter
        def sceneAlphaDepthPrepassInstanceVisits(self, arg: int, /) -> None: ...

        @property
        def sceneAlphaDepthPrepassFallbackDrawCalls(self) -> int: ...

        @sceneAlphaDepthPrepassFallbackDrawCalls.setter
        def sceneAlphaDepthPrepassFallbackDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneFrontToBackOrderedInstancedDrawCalls(self) -> int: ...

        @sceneFrontToBackOrderedInstancedDrawCalls.setter
        def sceneFrontToBackOrderedInstancedDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneFrontToBackOrderedFoliageDrawCalls(self) -> int: ...

        @sceneFrontToBackOrderedFoliageDrawCalls.setter
        def sceneFrontToBackOrderedFoliageDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneFrontToBackOrderingFallbackDrawCalls(self) -> int: ...

        @sceneFrontToBackOrderingFallbackDrawCalls.setter
        def sceneFrontToBackOrderingFallbackDrawCalls(self, arg: int, /) -> None: ...

        @property
        def maxSourceTrianglesPerInstance(self) -> int: ...

        @maxSourceTrianglesPerInstance.setter
        def maxSourceTrianglesPerInstance(self, arg: int, /) -> None: ...

        @property
        def maxImpostorTrianglesPerInstance(self) -> int: ...

        @maxImpostorTrianglesPerInstance.setter
        def maxImpostorTrianglesPerInstance(self, arg: int, /) -> None: ...

        @property
        def conservativeOcclusionCandidates(self) -> int: ...

        @conservativeOcclusionCandidates.setter
        def conservativeOcclusionCandidates(self, arg: int, /) -> None: ...

        @property
        def conservativeOcclusionQueries(self) -> int: ...

        @conservativeOcclusionQueries.setter
        def conservativeOcclusionQueries(self, arg: int, /) -> None: ...

        @property
        def conservativeOcclusionOccluded(self) -> int: ...

        @conservativeOcclusionOccluded.setter
        def conservativeOcclusionOccluded(self, arg: int, /) -> None: ...

        @property
        def conservativeOcclusionDrawn(self) -> int: ...

        @conservativeOcclusionDrawn.setter
        def conservativeOcclusionDrawn(self, arg: int, /) -> None: ...

        @property
        def hizOcclusionTests(self) -> int: ...

        @hizOcclusionTests.setter
        def hizOcclusionTests(self, arg: int, /) -> None: ...

        @property
        def hizOcclusionOccluded(self) -> int: ...

        @hizOcclusionOccluded.setter
        def hizOcclusionOccluded(self, arg: int, /) -> None: ...

        @property
        def hizOcclusionDepthReadbacks(self) -> int: ...

        @hizOcclusionDepthReadbacks.setter
        def hizOcclusionDepthReadbacks(self, arg: int, /) -> None: ...

        @property
        def gpuHizOcclusionDrawCalls(self) -> int: ...

        @gpuHizOcclusionDrawCalls.setter
        def gpuHizOcclusionDrawCalls(self, arg: int, /) -> None: ...

        @property
        def gpuHizOcclusionDepthBuilds(self) -> int: ...

        @gpuHizOcclusionDepthBuilds.setter
        def gpuHizOcclusionDepthBuilds(self, arg: int, /) -> None: ...

        @property
        def gpuHizOcclusionComputeDepthBuilds(self) -> int: ...

        @gpuHizOcclusionComputeDepthBuilds.setter
        def gpuHizOcclusionComputeDepthBuilds(self, arg: int, /) -> None: ...

        @property
        def gpuHizOcclusionRasterDepthBuilds(self) -> int: ...

        @gpuHizOcclusionRasterDepthBuilds.setter
        def gpuHizOcclusionRasterDepthBuilds(self, arg: int, /) -> None: ...

        @property
        def sceneGpuOcclusionFilteredDrawCalls(self) -> int: ...

        @sceneGpuOcclusionFilteredDrawCalls.setter
        def sceneGpuOcclusionFilteredDrawCalls(self, arg: int, /) -> None: ...

        @property
        def sceneFoliageGpuMs(self) -> float: ...

        @sceneFoliageGpuMs.setter
        def sceneFoliageGpuMs(self, arg: float, /) -> None: ...

        @property
        def shadowFoliageGpuMs(self) -> float: ...

        @shadowFoliageGpuMs.setter
        def shadowFoliageGpuMs(self, arg: float, /) -> None: ...

    class ImportVisualSceneOptions:
        def __init__(self) -> None: ...

        @property
        def importLights(self) -> bool: ...

        @importLights.setter
        def importLights(self, arg: bool, /) -> None: ...

        @property
        def lightIntensityScale(self) -> float: ...

        @lightIntensityScale.setter
        def lightIntensityScale(self, arg: float, /) -> None: ...

        @property
        def scale(self) -> numpy.ndarray: ...

        @scale.setter
        def scale(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def color(self) -> numpy.ndarray: ...

        @color.setter
        def color(self, arg: numpy.ndarray, /) -> None: ...

    class MaterialTextureBudgetRecommendation:
        def __init__(self) -> None: ...

        @property
        def materialTextureReferences(self) -> int: ...

        @materialTextureReferences.setter
        def materialTextureReferences(self, arg: int, /) -> None: ...

        @property
        def uniqueMaterialTextures(self) -> int: ...

        @uniqueMaterialTextures.setter
        def uniqueMaterialTextures(self, arg: int, /) -> None: ...

        @property
        def materialTextureBytes(self) -> int: ...

        @materialTextureBytes.setter
        def materialTextureBytes(self, arg: int, /) -> None: ...

        @property
        def budgetBytes(self) -> int: ...

        @budgetBytes.setter
        def budgetBytes(self, arg: int, /) -> None: ...

        @property
        def overBudget(self) -> bool: ...

        @overBudget.setter
        def overBudget(self, arg: bool, /) -> None: ...

        @property
        def changed(self) -> bool: ...

        @changed.setter
        def changed(self, arg: bool, /) -> None: ...

        @property
        def recommendedTextureAnisotropy(self) -> float: ...

        @recommendedTextureAnisotropy.setter
        def recommendedTextureAnisotropy(self, arg: float, /) -> None: ...

        @property
        def recommendedTextureMipLodBias(self) -> float: ...

        @recommendedTextureMipLodBias.setter
        def recommendedTextureMipLodBias(self, arg: float, /) -> None: ...

        @property
        def recommendedPbrEnvironmentMipLodBias(self) -> float: ...

        @recommendedPbrEnvironmentMipLodBias.setter
        def recommendedPbrEnvironmentMipLodBias(self, arg: float, /) -> None: ...

        @property
        def recommendedReflectionProbeFilteringEnabled(self) -> bool: ...

        @recommendedReflectionProbeFilteringEnabled.setter
        def recommendedReflectionProbeFilteringEnabled(self, arg: bool, /) -> None: ...

        @property
        def recommendedSettings(self) -> RenderQualitySettings: ...

        @recommendedSettings.setter
        def recommendedSettings(self, arg: RenderQualitySettings, /) -> None: ...

        @property
        def reason(self) -> str: ...

        @reason.setter
        def reason(self, arg: str, /) -> None: ...

    class ReflectionProbe:
        def __init__(self) -> None: ...

        @property
        def position(self) -> numpy.ndarray: ...

        @position.setter
        def position(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def environmentMap(self) -> int: ...

        @environmentMap.setter
        def environmentMap(self, arg: int, /) -> None: ...

        @property
        def irradianceMap(self) -> int: ...

        @irradianceMap.setter
        def irradianceMap(self, arg: int, /) -> None: ...

        @property
        def prefilteredEnvironmentMap(self) -> int: ...

        @prefilteredEnvironmentMap.setter
        def prefilteredEnvironmentMap(self, arg: int, /) -> None: ...

        @property
        def brdfLut(self) -> int: ...

        @brdfLut.setter
        def brdfLut(self, arg: int, /) -> None: ...

        @property
        def strength(self) -> float: ...

        @strength.setter
        def strength(self, arg: float, /) -> None: ...

        @property
        def boxProjection(self) -> bool: ...

        @boxProjection.setter
        def boxProjection(self, arg: bool, /) -> None: ...

        @property
        def boxMin(self) -> numpy.ndarray: ...

        @boxMin.setter
        def boxMin(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def boxMax(self) -> numpy.ndarray: ...

        @boxMax.setter
        def boxMax(self, arg: numpy.ndarray, /) -> None: ...

    class ReflectionProbeBlend:
        def __init__(self) -> None: ...

        @property
        def valid(self) -> bool: ...

        @valid.setter
        def valid(self, arg: bool, /) -> None: ...

        @property
        def primaryIndex(self) -> int: ...

        @primaryIndex.setter
        def primaryIndex(self, arg: int, /) -> None: ...

        @property
        def secondaryIndex(self) -> int: ...

        @secondaryIndex.setter
        def secondaryIndex(self, arg: int, /) -> None: ...

        @property
        def primaryWeight(self) -> float: ...

        @primaryWeight.setter
        def primaryWeight(self, arg: float, /) -> None: ...

        @property
        def secondaryWeight(self) -> float: ...

        @secondaryWeight.setter
        def secondaryWeight(self, arg: float, /) -> None: ...

        @property
        def strength(self) -> float: ...

        @strength.setter
        def strength(self, arg: float, /) -> None: ...

        @property
        def primaryBoxProjection(self) -> bool: ...

        @primaryBoxProjection.setter
        def primaryBoxProjection(self, arg: bool, /) -> None: ...

        @property
        def secondaryBoxProjection(self) -> bool: ...

        @secondaryBoxProjection.setter
        def secondaryBoxProjection(self, arg: bool, /) -> None: ...

        @property
        def primaryProbePosition(self) -> numpy.ndarray: ...

        @primaryProbePosition.setter
        def primaryProbePosition(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def secondaryProbePosition(self) -> numpy.ndarray: ...

        @secondaryProbePosition.setter
        def secondaryProbePosition(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def primaryBoxMin(self) -> numpy.ndarray: ...

        @primaryBoxMin.setter
        def primaryBoxMin(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def primaryBoxMax(self) -> numpy.ndarray: ...

        @primaryBoxMax.setter
        def primaryBoxMax(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def secondaryBoxMin(self) -> numpy.ndarray: ...

        @secondaryBoxMin.setter
        def secondaryBoxMin(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def secondaryBoxMax(self) -> numpy.ndarray: ...

        @secondaryBoxMax.setter
        def secondaryBoxMax(self, arg: numpy.ndarray, /) -> None: ...

    class ReflectionProbeCaptureFrame:
        def __init__(self) -> None: ...

        @property
        def direction(self) -> numpy.ndarray: ...

        @direction.setter
        def direction(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def up(self) -> numpy.ndarray: ...

        @up.setter
        def up(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def right(self) -> numpy.ndarray: ...

        @right.setter
        def right(self, arg: numpy.ndarray, /) -> None: ...

    class ReflectionProbeCaptureSettings:
        def __init__(self) -> None: ...

        @property
        def resolution(self) -> int: ...

        @resolution.setter
        def resolution(self, arg: int, /) -> None: ...

        @property
        def drawVisualizationObjects(self) -> bool: ...

        @drawVisualizationObjects.setter
        def drawVisualizationObjects(self, arg: bool, /) -> None: ...

        @property
        def drawCoordinateFrames(self) -> bool: ...

        @drawCoordinateFrames.setter
        def drawCoordinateFrames(self, arg: bool, /) -> None: ...

        @property
        def drawPointClouds(self) -> bool: ...

        @drawPointClouds.setter
        def drawPointClouds(self, arg: bool, /) -> None: ...

        @property
        def doShadows(self) -> bool: ...

        @doShadows.setter
        def doShadows(self, arg: bool, /) -> None: ...

        @property
        def includeEnvironmentBackground(self) -> bool: ...

        @includeEnvironmentBackground.setter
        def includeEnvironmentBackground(self, arg: bool, /) -> None: ...

        @property
        def environmentBackgroundMap(self) -> int: ...

        @environmentBackgroundMap.setter
        def environmentBackgroundMap(self, arg: int, /) -> None: ...

        @property
        def environmentBackgroundExposure(self) -> float: ...

        @environmentBackgroundExposure.setter
        def environmentBackgroundExposure(self, arg: float, /) -> None: ...

    class ReflectionProbeDebugItem:
        def __init__(self) -> None: ...

        @property
        def index(self) -> int: ...

        @index.setter
        def index(self, arg: int, /) -> None: ...

        @property
        def selected(self) -> bool: ...

        @selected.setter
        def selected(self, arg: bool, /) -> None: ...

        @property
        def primary(self) -> bool: ...

        @primary.setter
        def primary(self, arg: bool, /) -> None: ...

        @property
        def secondary(self) -> bool: ...

        @secondary.setter
        def secondary(self, arg: bool, /) -> None: ...

        @property
        def rawWeight(self) -> float: ...

        @rawWeight.setter
        def rawWeight(self, arg: float, /) -> None: ...

        @property
        def normalizedWeight(self) -> float: ...

        @normalizedWeight.setter
        def normalizedWeight(self, arg: float, /) -> None: ...

        @property
        def position(self) -> numpy.ndarray: ...

        @position.setter
        def position(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def boxProjection(self) -> bool: ...

        @boxProjection.setter
        def boxProjection(self, arg: bool, /) -> None: ...

        @property
        def boxProjectionValid(self) -> bool: ...

        @boxProjectionValid.setter
        def boxProjectionValid(self, arg: bool, /) -> None: ...

        @property
        def boxMin(self) -> numpy.ndarray: ...

        @boxMin.setter
        def boxMin(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def boxMax(self) -> numpy.ndarray: ...

        @boxMax.setter
        def boxMax(self, arg: numpy.ndarray, /) -> None: ...

    class ReflectionProbeDebugOverlaySummary:
        def __init__(self) -> None: ...

        @property
        def available(self) -> bool: ...

        @available.setter
        def available(self, arg: bool, /) -> None: ...

        @property
        def overlayEnabled(self) -> bool: ...

        @overlayEnabled.setter
        def overlayEnabled(self, arg: bool, /) -> None: ...

        @property
        def validBlend(self) -> bool: ...

        @validBlend.setter
        def validBlend(self, arg: bool, /) -> None: ...

        @property
        def registeredProbeCount(self) -> int: ...

        @registeredProbeCount.setter
        def registeredProbeCount(self, arg: int, /) -> None: ...

        @property
        def selectedProbeCount(self) -> int: ...

        @selectedProbeCount.setter
        def selectedProbeCount(self, arg: int, /) -> None: ...

        @property
        def boxProjectionProbeCount(self) -> int: ...

        @boxProjectionProbeCount.setter
        def boxProjectionProbeCount(self, arg: int, /) -> None: ...

        @property
        def validBoxProjectionProbeCount(self) -> int: ...

        @validBoxProjectionProbeCount.setter
        def validBoxProjectionProbeCount(self, arg: int, /) -> None: ...

        @property
        def primaryIndex(self) -> int: ...

        @primaryIndex.setter
        def primaryIndex(self, arg: int, /) -> None: ...

        @property
        def secondaryIndex(self) -> int: ...

        @secondaryIndex.setter
        def secondaryIndex(self, arg: int, /) -> None: ...

        @property
        def primaryWeight(self) -> float: ...

        @primaryWeight.setter
        def primaryWeight(self, arg: float, /) -> None: ...

        @property
        def secondaryWeight(self) -> float: ...

        @secondaryWeight.setter
        def secondaryWeight(self, arg: float, /) -> None: ...

        @property
        def strength(self) -> float: ...

        @strength.setter
        def strength(self, arg: float, /) -> None: ...

        @property
        def samplePosition(self) -> numpy.ndarray: ...

        @samplePosition.setter
        def samplePosition(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def primaryPosition(self) -> numpy.ndarray: ...

        @primaryPosition.setter
        def primaryPosition(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def secondaryPosition(self) -> numpy.ndarray: ...

        @secondaryPosition.setter
        def secondaryPosition(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def primaryBoxProjection(self) -> bool: ...

        @primaryBoxProjection.setter
        def primaryBoxProjection(self, arg: bool, /) -> None: ...

        @property
        def secondaryBoxProjection(self) -> bool: ...

        @secondaryBoxProjection.setter
        def secondaryBoxProjection(self, arg: bool, /) -> None: ...

    class ReflectionProbeDebugView:
        def __init__(self) -> None: ...

        @property
        def valid(self) -> bool: ...

        @valid.setter
        def valid(self, arg: bool, /) -> None: ...

        @property
        def samplePosition(self) -> numpy.ndarray: ...

        @samplePosition.setter
        def samplePosition(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def blend(self) -> RayraiWindow.ReflectionProbeBlend: ...

        @blend.setter
        def blend(self, arg: RayraiWindow.ReflectionProbeBlend, /) -> None: ...

        @property
        def probes(self) -> list[RayraiWindow.ReflectionProbeDebugItem]: ...

        @probes.setter
        def probes(self, arg: Sequence[RayraiWindow.ReflectionProbeDebugItem], /) -> None: ...

    class ReflectionProbeFilterSettings:
        def __init__(self) -> None: ...

        @property
        def irradianceResolution(self) -> int: ...

        @irradianceResolution.setter
        def irradianceResolution(self, arg: int, /) -> None: ...

        @property
        def irradianceSamples(self) -> int: ...

        @irradianceSamples.setter
        def irradianceSamples(self, arg: int, /) -> None: ...

        @property
        def prefilteredResolution(self) -> int: ...

        @prefilteredResolution.setter
        def prefilteredResolution(self, arg: int, /) -> None: ...

        @property
        def prefilteredMipLevels(self) -> int: ...

        @prefilteredMipLevels.setter
        def prefilteredMipLevels(self, arg: int, /) -> None: ...

        @property
        def prefilteredSamples(self) -> int: ...

        @prefilteredSamples.setter
        def prefilteredSamples(self, arg: int, /) -> None: ...

        @property
        def brdfLutSize(self) -> int: ...

        @brdfLutSize.setter
        def brdfLutSize(self, arg: int, /) -> None: ...

        @property
        def brdfLutSamples(self) -> int: ...

        @brdfLutSamples.setter
        def brdfLutSamples(self, arg: int, /) -> None: ...

    class ReflectionProbePlacementSuggestion:
        def __init__(self) -> None: ...

        @property
        def valid(self) -> bool: ...

        @valid.setter
        def valid(self, arg: bool, /) -> None: ...

        @property
        def position(self) -> numpy.ndarray: ...

        @position.setter
        def position(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def boxProjection(self) -> bool: ...

        @boxProjection.setter
        def boxProjection(self, arg: bool, /) -> None: ...

        @property
        def boxMin(self) -> numpy.ndarray: ...

        @boxMin.setter
        def boxMin(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def boxMax(self) -> numpy.ndarray: ...

        @boxMax.setter
        def boxMax(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def boundedObjectCount(self) -> int: ...

        @boundedObjectCount.setter
        def boundedObjectCount(self, arg: int, /) -> None: ...

        @property
        def boundsDiagonal(self) -> float: ...

        @boundsDiagonal.setter
        def boundsDiagonal(self, arg: float, /) -> None: ...

    class ReflectionProbeSidecarLoadResult:
        def __init__(self) -> None: ...

        @property
        def probes(self) -> list[RayraiWindow.ReflectionProbeSidecarProbe]: ...

        @probes.setter
        def probes(self, arg: Sequence[RayraiWindow.ReflectionProbeSidecarProbe], /) -> None: ...

        @property
        def settings(self) -> RayraiWindow.ReflectionProbeSidecarSettings: ...

        @settings.setter
        def settings(self, arg: RayraiWindow.ReflectionProbeSidecarSettings, /) -> None: ...

        @property
        def loadedSidecarPath(self) -> str: ...

        @loadedSidecarPath.setter
        def loadedSidecarPath(self, arg: str, /) -> None: ...

    class ReflectionProbeSidecarProbe:
        def __init__(self) -> None: ...

        @property
        def name(self) -> str: ...

        @name.setter
        def name(self, arg: str, /) -> None: ...

        @property
        def position(self) -> numpy.ndarray: ...

        @position.setter
        def position(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def strength(self) -> float: ...

        @strength.setter
        def strength(self, arg: float, /) -> None: ...

        @property
        def boxProjection(self) -> bool: ...

        @boxProjection.setter
        def boxProjection(self, arg: bool, /) -> None: ...

        @property
        def boxMin(self) -> numpy.ndarray: ...

        @boxMin.setter
        def boxMin(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def boxMax(self) -> numpy.ndarray: ...

        @boxMax.setter
        def boxMax(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def capture(self) -> RayraiWindow.ReflectionProbeCaptureSettings: ...

        @capture.setter
        def capture(self, arg: RayraiWindow.ReflectionProbeCaptureSettings, /) -> None: ...

        @property
        def hasFilter(self) -> bool: ...

        @hasFilter.setter
        def hasFilter(self, arg: bool, /) -> None: ...

        @property
        def filter(self) -> RayraiWindow.ReflectionProbeFilterSettings: ...

        @filter.setter
        def filter(self, arg: RayraiWindow.ReflectionProbeFilterSettings, /) -> None: ...

    class ReflectionProbeSidecarSettings:
        def __init__(self) -> None: ...

        @property
        def hasMaxReflectionProbesPerSelection(self) -> bool: ...

        @hasMaxReflectionProbesPerSelection.setter
        def hasMaxReflectionProbesPerSelection(self, arg: bool, /) -> None: ...

        @property
        def maxReflectionProbesPerSelection(self) -> int: ...

        @maxReflectionProbesPerSelection.setter
        def maxReflectionProbesPerSelection(self, arg: int, /) -> None: ...

        @property
        def hasMinReflectionProbeWeight(self) -> bool: ...

        @hasMinReflectionProbeWeight.setter
        def hasMinReflectionProbeWeight(self, arg: bool, /) -> None: ...

        @property
        def minReflectionProbeWeight(self) -> float: ...

        @minReflectionProbeWeight.setter
        def minReflectionProbeWeight(self, arg: float, /) -> None: ...

        @property
        def hasReflectionProbeFilteringEnabled(self) -> bool: ...

        @hasReflectionProbeFilteringEnabled.setter
        def hasReflectionProbeFilteringEnabled(self, arg: bool, /) -> None: ...

        @property
        def reflectionProbeFilteringEnabled(self) -> bool: ...

        @reflectionProbeFilteringEnabled.setter
        def reflectionProbeFilteringEnabled(self, arg: bool, /) -> None: ...

        @property
        def hasReflectionProbeFilter(self) -> bool: ...

        @hasReflectionProbeFilter.setter
        def hasReflectionProbeFilter(self, arg: bool, /) -> None: ...

        @property
        def reflectionProbeFilter(self) -> RayraiWindow.ReflectionProbeFilterSettings: ...

        @reflectionProbeFilter.setter
        def reflectionProbeFilter(self, arg: RayraiWindow.ReflectionProbeFilterSettings, /) -> None: ...

    class RenderOverrides:
        def __init__(self) -> None: ...

        @property
        def mesh(self) -> Shader | None: ...

        @mesh.setter
        def mesh(self, value: Shader | None) -> None: ...

        @property
        def pointcloud(self) -> Shader | None: ...

        @pointcloud.setter
        def pointcloud(self, value: Shader | None) -> None: ...

        @property
        def picking(self) -> Shader | None: ...

        @picking.setter
        def picking(self, value: Shader | None) -> None: ...

        @property
        def post(self) -> Shader | None: ...

        @post.setter
        def post(self, value: Shader | None) -> None: ...

        @property
        def doShadows(self) -> bool: ...

        @doShadows.setter
        def doShadows(self, arg: bool, /) -> None: ...

        @property
        def drawCoordinateFrames(self) -> bool: ...

        @drawCoordinateFrames.setter
        def drawCoordinateFrames(self, arg: bool, /) -> None: ...

        @property
        def drawPointClouds(self) -> bool: ...

        @drawPointClouds.setter
        def drawPointClouds(self, arg: bool, /) -> None: ...

        @property
        def drawVisualizationObjects(self) -> bool: ...

        @drawVisualizationObjects.setter
        def drawVisualizationObjects(self, arg: bool, /) -> None: ...

        @property
        def drawWeatherEffects(self) -> bool: ...

        @drawWeatherEffects.setter
        def drawWeatherEffects(self, arg: bool, /) -> None: ...

        @property
        def skipGroundObjects(self) -> bool: ...

        @skipGroundObjects.setter
        def skipGroundObjects(self, arg: bool, /) -> None: ...

        @property
        def environmentBackgroundMap(self) -> int: ...

        @environmentBackgroundMap.setter
        def environmentBackgroundMap(self, arg: int, /) -> None: ...

        @property
        def environmentBackgroundExposure(self) -> float: ...

        @environmentBackgroundExposure.setter
        def environmentBackgroundExposure(self, arg: float, /) -> None: ...

        @property
        def postProcess(self) -> bool: ...

        @postProcess.setter
        def postProcess(self, arg: bool, /) -> None: ...

        @property
        def allowViewerMsaa(self) -> bool: ...

        @allowViewerMsaa.setter
        def allowViewerMsaa(self, arg: bool, /) -> None: ...

        @property
        def allowTemporalAa(self) -> bool: ...

        @allowTemporalAa.setter
        def allowTemporalAa(self, arg: bool, /) -> None: ...

        @property
        def allowViewerUpscale(self) -> bool: ...

        @allowViewerUpscale.setter
        def allowViewerUpscale(self, arg: bool, /) -> None: ...

        @property
        def fixedShadowCenterEnabled(self) -> bool: ...

        @fixedShadowCenterEnabled.setter
        def fixedShadowCenterEnabled(self, arg: bool, /) -> None: ...

        @property
        def fixedShadowCenter(self) -> numpy.ndarray: ...

        @fixedShadowCenter.setter
        def fixedShadowCenter(self, arg: numpy.ndarray, /) -> None: ...

    class RenderPassAccounting:
        def __init__(self) -> None: ...

        @property
        def fastForwardPathEligible(self) -> bool: ...

        @fastForwardPathEligible.setter
        def fastForwardPathEligible(self, arg: bool, /) -> None: ...

        @property
        def postprocessActive(self) -> bool: ...

        @postprocessActive.setter
        def postprocessActive(self, arg: bool, /) -> None: ...

        @property
        def msaaActive(self) -> bool: ...

        @msaaActive.setter
        def msaaActive(self, arg: bool, /) -> None: ...

        @property
        def renderScaleActive(self) -> bool: ...

        @renderScaleActive.setter
        def renderScaleActive(self, arg: bool, /) -> None: ...

        @property
        def probeFilteringActive(self) -> bool: ...

        @probeFilteringActive.setter
        def probeFilteringActive(self, arg: bool, /) -> None: ...

        @property
        def localProbeSelectionActive(self) -> bool: ...

        @localProbeSelectionActive.setter
        def localProbeSelectionActive(self, arg: bool, /) -> None: ...

        @property
        def additionalShadowActive(self) -> bool: ...

        @additionalShadowActive.setter
        def additionalShadowActive(self, arg: bool, /) -> None: ...

        @property
        def mainShadowPassesPerFrame(self) -> int: ...

        @mainShadowPassesPerFrame.setter
        def mainShadowPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def additional2dShadowPassesPerFrame(self) -> int: ...

        @additional2dShadowPassesPerFrame.setter
        def additional2dShadowPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def additionalPointShadowFacePassesPerFrame(self) -> int: ...

        @additionalPointShadowFacePassesPerFrame.setter
        def additionalPointShadowFacePassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def shadowPassesPerFrame(self) -> int: ...

        @shadowPassesPerFrame.setter
        def shadowPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def sceneColorPassesPerFrame(self) -> int: ...

        @sceneColorPassesPerFrame.setter
        def sceneColorPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def planarReflectionPassesPerFrame(self) -> int: ...

        @planarReflectionPassesPerFrame.setter
        def planarReflectionPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def renderScaleResolvePassesPerFrame(self) -> int: ...

        @renderScaleResolvePassesPerFrame.setter
        def renderScaleResolvePassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def transparentPassesPerFrame(self) -> int: ...

        @transparentPassesPerFrame.setter
        def transparentPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def postprocessPassesPerFrame(self) -> int: ...

        @postprocessPassesPerFrame.setter
        def postprocessPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def bloomExtraPassesPerFrame(self) -> int: ...

        @bloomExtraPassesPerFrame.setter
        def bloomExtraPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def aoDenoisePassesPerFrame(self) -> int: ...

        @aoDenoisePassesPerFrame.setter
        def aoDenoisePassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def debugReadbackPassesPerFrame(self) -> int: ...

        @debugReadbackPassesPerFrame.setter
        def debugReadbackPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def totalEstimatedPassesPerFrame(self) -> int: ...

        @totalEstimatedPassesPerFrame.setter
        def totalEstimatedPassesPerFrame(self, arg: int, /) -> None: ...

        @property
        def activeAdditionalLights(self) -> int: ...

        @activeAdditionalLights.setter
        def activeAdditionalLights(self, arg: int, /) -> None: ...

        @property
        def activeAdditionalShadowLights(self) -> int: ...

        @activeAdditionalShadowLights.setter
        def activeAdditionalShadowLights(self, arg: int, /) -> None: ...

        @property
        def activeReflectionProbes(self) -> int: ...

        @activeReflectionProbes.setter
        def activeReflectionProbes(self, arg: int, /) -> None: ...

        @property
        def activeUltraFeatureCount(self) -> int: ...

        @activeUltraFeatureCount.setter
        def activeUltraFeatureCount(self, arg: int, /) -> None: ...

        @property
        def likelyBottleneck(self) -> str: ...

        @likelyBottleneck.setter
        def likelyBottleneck(self, arg: str, /) -> None: ...

    class RenderPassTiming:
        def __init__(self) -> None: ...

        @property
        def name(self) -> str: ...

        @name.setter
        def name(self, arg: str, /) -> None: ...

        @property
        def gpuMs(self) -> float: ...

        @gpuMs.setter
        def gpuMs(self, arg: float, /) -> None: ...

    class RgbaCalibrationMetrics:
        def __init__(self) -> None: ...

        @property
        def width(self) -> int: ...

        @width.setter
        def width(self, arg: int, /) -> None: ...

        @property
        def height(self) -> int: ...

        @height.setter
        def height(self, arg: int, /) -> None: ...

        @property
        def patchCount(self) -> int: ...

        @patchCount.setter
        def patchCount(self, arg: int, /) -> None: ...

        @property
        def validPatchCount(self) -> int: ...

        @validPatchCount.setter
        def validPatchCount(self, arg: int, /) -> None: ...

        @property
        def averageRgbError(self) -> float: ...

        @averageRgbError.setter
        def averageRgbError(self, arg: float, /) -> None: ...

        @property
        def rmsRgbError(self) -> float: ...

        @rmsRgbError.setter
        def rmsRgbError(self, arg: float, /) -> None: ...

        @property
        def maxRgbError(self) -> float: ...

        @maxRgbError.setter
        def maxRgbError(self, arg: float, /) -> None: ...

        @property
        def averageLuminanceError(self) -> float: ...

        @averageLuminanceError.setter
        def averageLuminanceError(self, arg: float, /) -> None: ...

        @property
        def averageSaturationError(self) -> float: ...

        @averageSaturationError.setter
        def averageSaturationError(self, arg: float, /) -> None: ...

        @property
        def patches(self) -> list[RayraiWindow.RgbaCalibrationPatchResult]: ...

        @patches.setter
        def patches(self, arg: Sequence[RayraiWindow.RgbaCalibrationPatchResult], /) -> None: ...

    class RgbaCalibrationPatch:
        def __init__(self) -> None: ...

        @property
        def name(self) -> str: ...

        @name.setter
        def name(self, arg: str, /) -> None: ...

        @property
        def x(self) -> int: ...

        @x.setter
        def x(self, arg: int, /) -> None: ...

        @property
        def y(self) -> int: ...

        @y.setter
        def y(self, arg: int, /) -> None: ...

        @property
        def width(self) -> int: ...

        @width.setter
        def width(self, arg: int, /) -> None: ...

        @property
        def height(self) -> int: ...

        @height.setter
        def height(self, arg: int, /) -> None: ...

        @property
        def targetRgb(self) -> numpy.ndarray: ...

        @targetRgb.setter
        def targetRgb(self, arg: numpy.ndarray, /) -> None: ...

    class RgbaCalibrationPatchResult:
        def __init__(self) -> None: ...

        @property
        def name(self) -> str: ...

        @name.setter
        def name(self, arg: str, /) -> None: ...

        @property
        def valid(self) -> bool: ...

        @valid.setter
        def valid(self, arg: bool, /) -> None: ...

        @property
        def sampleCount(self) -> int: ...

        @sampleCount.setter
        def sampleCount(self, arg: int, /) -> None: ...

        @property
        def measuredRgb(self) -> numpy.ndarray: ...

        @measuredRgb.setter
        def measuredRgb(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def targetRgb(self) -> numpy.ndarray: ...

        @targetRgb.setter
        def targetRgb(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def rgbError(self) -> float: ...

        @rgbError.setter
        def rgbError(self, arg: float, /) -> None: ...

        @property
        def maxChannelError(self) -> float: ...

        @maxChannelError.setter
        def maxChannelError(self, arg: float, /) -> None: ...

        @property
        def luminanceError(self) -> float: ...

        @luminanceError.setter
        def luminanceError(self, arg: float, /) -> None: ...

        @property
        def saturationError(self) -> float: ...

        @saturationError.setter
        def saturationError(self, arg: float, /) -> None: ...

    class RgbaCalibrationRecommendation:
        def __init__(self) -> None: ...

        @property
        def valid(self) -> bool: ...

        @valid.setter
        def valid(self, arg: bool, /) -> None: ...

        @property
        def changed(self) -> bool: ...

        @changed.setter
        def changed(self, arg: bool, /) -> None: ...

        @property
        def neutralPatchCount(self) -> int: ...

        @neutralPatchCount.setter
        def neutralPatchCount(self, arg: int, /) -> None: ...

        @property
        def saturatedPatchCount(self) -> int: ...

        @saturatedPatchCount.setter
        def saturatedPatchCount(self, arg: int, /) -> None: ...

        @property
        def averageRgbErrorBefore(self) -> float: ...

        @averageRgbErrorBefore.setter
        def averageRgbErrorBefore(self, arg: float, /) -> None: ...

        @property
        def outputTransform(self) -> RayraiWindow.RgbaOutputTransform: ...

        @outputTransform.setter
        def outputTransform(self, arg: RayraiWindow.RgbaOutputTransform, /) -> None: ...

        @property
        def reason(self) -> str: ...

        @reason.setter
        def reason(self, arg: str, /) -> None: ...

    class RgbaLuminanceHistogram:
        def __init__(self) -> None: ...

        @property
        def width(self) -> int: ...

        @width.setter
        def width(self, arg: int, /) -> None: ...

        @property
        def height(self) -> int: ...

        @height.setter
        def height(self, arg: int, /) -> None: ...

        @property
        def pixelCount(self) -> int: ...

        @pixelCount.setter
        def pixelCount(self, arg: int, /) -> None: ...

        @property
        def binCount(self) -> int: ...

        @binCount.setter
        def binCount(self, arg: int, /) -> None: ...

        @property
        def peakBin(self) -> int: ...

        @peakBin.setter
        def peakBin(self, arg: int, /) -> None: ...

        @property
        def peakFraction(self) -> float: ...

        @peakFraction.setter
        def peakFraction(self, arg: float, /) -> None: ...

        @property
        def bins(self) -> list[int]: ...

        @bins.setter
        def bins(self, arg: Sequence[int], /) -> None: ...

    class RgbaLuminanceMetrics:
        def __init__(self) -> None: ...

        @property
        def width(self) -> int: ...

        @width.setter
        def width(self, arg: int, /) -> None: ...

        @property
        def height(self) -> int: ...

        @height.setter
        def height(self, arg: int, /) -> None: ...

        @property
        def pixelCount(self) -> int: ...

        @pixelCount.setter
        def pixelCount(self, arg: int, /) -> None: ...

        @property
        def averageLuminance(self) -> float: ...

        @averageLuminance.setter
        def averageLuminance(self, arg: float, /) -> None: ...

        @property
        def medianLuminance(self) -> float: ...

        @medianLuminance.setter
        def medianLuminance(self, arg: float, /) -> None: ...

        @property
        def p95Luminance(self) -> float: ...

        @p95Luminance.setter
        def p95Luminance(self, arg: float, /) -> None: ...

        @property
        def p99Luminance(self) -> float: ...

        @p99Luminance.setter
        def p99Luminance(self, arg: float, /) -> None: ...

        @property
        def minLuminance(self) -> float: ...

        @minLuminance.setter
        def minLuminance(self, arg: float, /) -> None: ...

        @property
        def maxLuminance(self) -> float: ...

        @maxLuminance.setter
        def maxLuminance(self, arg: float, /) -> None: ...

        @property
        def averageSaturation(self) -> float: ...

        @averageSaturation.setter
        def averageSaturation(self, arg: float, /) -> None: ...

        @property
        def contrastP95ToMedian(self) -> float: ...

        @contrastP95ToMedian.setter
        def contrastP95ToMedian(self, arg: float, /) -> None: ...

        @property
        def clippedChannelFraction(self) -> float: ...

        @clippedChannelFraction.setter
        def clippedChannelFraction(self, arg: float, /) -> None: ...

    class RgbaOutputTransform:
        def __init__(self) -> None: ...

        @property
        def exposure(self) -> float: ...

        @exposure.setter
        def exposure(self, arg: float, /) -> None: ...

        @property
        def whiteBalance(self) -> numpy.ndarray: ...

        @whiteBalance.setter
        def whiteBalance(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def saturation(self) -> float: ...

        @saturation.setter
        def saturation(self, arg: float, /) -> None: ...

        @property
        def gamma(self) -> float: ...

        @gamma.setter
        def gamma(self, arg: float, /) -> None: ...

    class SceneImportReport:
        def __init__(self) -> None: ...

        @property
        def visualLoaded(self) -> bool: ...

        @visualLoaded.setter
        def visualLoaded(self, arg: bool, /) -> None: ...

        @property
        def totalSceneLightCount(self) -> int: ...

        @totalSceneLightCount.setter
        def totalSceneLightCount(self, arg: int, /) -> None: ...

        @property
        def importedLightCount(self) -> int: ...

        @importedLightCount.setter
        def importedLightCount(self, arg: int, /) -> None: ...

        @property
        def droppedLightCount(self) -> int: ...

        @droppedLightCount.setter
        def droppedLightCount(self, arg: int, /) -> None: ...

        @property
        def shadowLightPromoted(self) -> bool: ...

        @shadowLightPromoted.setter
        def shadowLightPromoted(self, arg: bool, /) -> None: ...

        @property
        def remainingShadingLightCount(self) -> int: ...

        @remainingShadingLightCount.setter
        def remainingShadingLightCount(self, arg: int, /) -> None: ...

        @property
        def notes(self) -> str: ...

        @notes.setter
        def notes(self, arg: str, /) -> None: ...

    class ShaderWarmupDiagnostics:
        def __init__(self) -> None: ...

        @property
        def shaderProgramCount(self) -> int: ...

        @shaderProgramCount.setter
        def shaderProgramCount(self, arg: int, /) -> None: ...

        @property
        def linkedProgramCount(self) -> int: ...

        @linkedProgramCount.setter
        def linkedProgramCount(self, arg: int, /) -> None: ...

        @property
        def totalCompileMs(self) -> float: ...

        @totalCompileMs.setter
        def totalCompileMs(self, arg: float, /) -> None: ...

        @property
        def slowestCompileMs(self) -> float: ...

        @slowestCompileMs.setter
        def slowestCompileMs(self, arg: float, /) -> None: ...

        @property
        def slowestShaderName(self) -> str: ...

        @slowestShaderName.setter
        def slowestShaderName(self, arg: str, /) -> None: ...

        @property
        def warmedUp(self) -> bool: ...

        @warmedUp.setter
        def warmedUp(self, arg: bool, /) -> None: ...

    class ShadowAtlasPlan:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def overflow(self) -> bool: ...

        @overflow.setter
        def overflow(self, arg: bool, /) -> None: ...

        @property
        def pointShadowPolicy(self) -> PointShadowPolicy: ...

        @pointShadowPolicy.setter
        def pointShadowPolicy(self, arg: PointShadowPolicy, /) -> None: ...

        @property
        def pointShadowsUseStandaloneCubemaps(self) -> bool: ...

        @pointShadowsUseStandaloneCubemaps.setter
        def pointShadowsUseStandaloneCubemaps(self, arg: bool, /) -> None: ...

        @property
        def atlasResolution(self) -> int: ...

        @atlasResolution.setter
        def atlasResolution(self, arg: int, /) -> None: ...

        @property
        def padding(self) -> int: ...

        @padding.setter
        def padding(self, arg: int, /) -> None: ...

        @property
        def tileCount(self) -> int: ...

        @tileCount.setter
        def tileCount(self, arg: int, /) -> None: ...

        @property
        def nonPointTileCount(self) -> int: ...

        @nonPointTileCount.setter
        def nonPointTileCount(self, arg: int, /) -> None: ...

        @property
        def pointShadowFaceTileCount(self) -> int: ...

        @pointShadowFaceTileCount.setter
        def pointShadowFaceTileCount(self, arg: int, /) -> None: ...

        @property
        def occupancy(self) -> float: ...

        @occupancy.setter
        def occupancy(self, arg: float, /) -> None: ...

        @property
        def tiles(self) -> list[RayraiWindow.ShadowAtlasTile]: ...

        @tiles.setter
        def tiles(self, arg: Sequence[RayraiWindow.ShadowAtlasTile], /) -> None: ...

    class ShadowAtlasTile:
        def __init__(self) -> None: ...

        @property
        def shadowIndex(self) -> int: ...

        @shadowIndex.setter
        def shadowIndex(self, arg: int, /) -> None: ...

        @property
        def sourceLightIndex(self) -> int: ...

        @sourceLightIndex.setter
        def sourceLightIndex(self, arg: int, /) -> None: ...

        @property
        def face(self) -> int: ...

        @face.setter
        def face(self, arg: int, /) -> None: ...

        @property
        def pointShadow(self) -> bool: ...

        @pointShadow.setter
        def pointShadow(self, arg: bool, /) -> None: ...

        @property
        def resolution(self) -> int: ...

        @resolution.setter
        def resolution(self, arg: int, /) -> None: ...

        @property
        def x(self) -> int: ...

        @x.setter
        def x(self, arg: int, /) -> None: ...

        @property
        def y(self) -> int: ...

        @y.setter
        def y(self, arg: int, /) -> None: ...

        @property
        def w(self) -> int: ...

        @w.setter
        def w(self, arg: int, /) -> None: ...

        @property
        def h(self) -> int: ...

        @h.setter
        def h(self, arg: int, /) -> None: ...

        @property
        def uvRect(self) -> numpy.ndarray: ...

        @uvRect.setter
        def uvRect(self, arg: numpy.ndarray, /) -> None: ...

    class ShadowCacheDiagnostics:
        def __init__(self) -> None: ...

        @property
        def available(self) -> bool: ...

        @available.setter
        def available(self, arg: bool, /) -> None: ...

        @property
        def eligible(self) -> bool: ...

        @eligible.setter
        def eligible(self, arg: bool, /) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def fallback(self) -> bool: ...

        @fallback.setter
        def fallback(self, arg: bool, /) -> None: ...

        @property
        def shadowMapRenders(self) -> int: ...

        @shadowMapRenders.setter
        def shadowMapRenders(self, arg: int, /) -> None: ...

        @property
        def shadowMapReuses(self) -> int: ...

        @shadowMapReuses.setter
        def shadowMapReuses(self, arg: int, /) -> None: ...

        @property
        def fallbackFrames(self) -> int: ...

        @fallbackFrames.setter
        def fallbackFrames(self, arg: int, /) -> None: ...

        @property
        def perLightDirtyTrackingAvailable(self) -> bool: ...

        @perLightDirtyTrackingAvailable.setter
        def perLightDirtyTrackingAvailable(self, arg: bool, /) -> None: ...

        @property
        def perLightDirtyTrackingEnabled(self) -> bool: ...

        @perLightDirtyTrackingEnabled.setter
        def perLightDirtyTrackingEnabled(self, arg: bool, /) -> None: ...

        @property
        def perLightDirtyTrackingFallback(self) -> bool: ...

        @perLightDirtyTrackingFallback.setter
        def perLightDirtyTrackingFallback(self, arg: bool, /) -> None: ...

        @property
        def perLightDirtyTrackingFallbackFrames(self) -> int: ...

        @perLightDirtyTrackingFallbackFrames.setter
        def perLightDirtyTrackingFallbackFrames(self, arg: int, /) -> None: ...

        @property
        def additionalShadowMapRenders(self) -> int: ...

        @additionalShadowMapRenders.setter
        def additionalShadowMapRenders(self, arg: int, /) -> None: ...

        @property
        def additionalShadowMapReuses(self) -> int: ...

        @additionalShadowMapReuses.setter
        def additionalShadowMapReuses(self, arg: int, /) -> None: ...

        @property
        def additionalShadowDirtyLights(self) -> int: ...

        @additionalShadowDirtyLights.setter
        def additionalShadowDirtyLights(self, arg: int, /) -> None: ...

        @property
        def additionalShadowCleanLights(self) -> int: ...

        @additionalShadowCleanLights.setter
        def additionalShadowCleanLights(self, arg: int, /) -> None: ...

        @property
        def staticDynamicCompositionAvailable(self) -> bool: ...

        @staticDynamicCompositionAvailable.setter
        def staticDynamicCompositionAvailable(self, arg: bool, /) -> None: ...

        @property
        def staticDynamicCompositionEnabled(self) -> bool: ...

        @staticDynamicCompositionEnabled.setter
        def staticDynamicCompositionEnabled(self, arg: bool, /) -> None: ...

        @property
        def staticDynamicCompositionFallback(self) -> bool: ...

        @staticDynamicCompositionFallback.setter
        def staticDynamicCompositionFallback(self, arg: bool, /) -> None: ...

        @property
        def staticDynamicCompositionFallbackFrames(self) -> int: ...

        @staticDynamicCompositionFallbackFrames.setter
        def staticDynamicCompositionFallbackFrames(self, arg: int, /) -> None: ...

        @property
        def staticShadowMapRenders(self) -> int: ...

        @staticShadowMapRenders.setter
        def staticShadowMapRenders(self, arg: int, /) -> None: ...

        @property
        def staticShadowMapReuses(self) -> int: ...

        @staticShadowMapReuses.setter
        def staticShadowMapReuses(self, arg: int, /) -> None: ...

        @property
        def dynamicShadowMapRenders(self) -> int: ...

        @dynamicShadowMapRenders.setter
        def dynamicShadowMapRenders(self, arg: int, /) -> None: ...

        @property
        def staticDynamicCompositedLayers(self) -> int: ...

        @staticDynamicCompositedLayers.setter
        def staticDynamicCompositedLayers(self, arg: int, /) -> None: ...

        @property
        def additionalStaticShadowMapRenders(self) -> int: ...

        @additionalStaticShadowMapRenders.setter
        def additionalStaticShadowMapRenders(self, arg: int, /) -> None: ...

        @property
        def additionalStaticShadowMapReuses(self) -> int: ...

        @additionalStaticShadowMapReuses.setter
        def additionalStaticShadowMapReuses(self, arg: int, /) -> None: ...

        @property
        def additionalDynamicShadowMapRenders(self) -> int: ...

        @additionalDynamicShadowMapRenders.setter
        def additionalDynamicShadowMapRenders(self, arg: int, /) -> None: ...

        @property
        def additionalStaticDynamicCompositedTiles(self) -> int: ...

        @additionalStaticDynamicCompositedTiles.setter
        def additionalStaticDynamicCompositedTiles(self, arg: int, /) -> None: ...

        @property
        def staticShadowCasterCount(self) -> int: ...

        @staticShadowCasterCount.setter
        def staticShadowCasterCount(self, arg: int, /) -> None: ...

        @property
        def dynamicShadowCasterCount(self) -> int: ...

        @dynamicShadowCasterCount.setter
        def dynamicShadowCasterCount(self, arg: int, /) -> None: ...

        @property
        def receiverDepthFilteredMaps(self) -> int: ...

        @receiverDepthFilteredMaps.setter
        def receiverDepthFilteredMaps(self, arg: int, /) -> None: ...

    class ShadowDebugOverlaySummary:
        def __init__(self) -> None: ...

        @property
        def available(self) -> bool: ...

        @available.setter
        def available(self, arg: bool, /) -> None: ...

        @property
        def overlayEnabled(self) -> bool: ...

        @overlayEnabled.setter
        def overlayEnabled(self, arg: bool, /) -> None: ...

        @property
        def shadowsEnabled(self) -> bool: ...

        @shadowsEnabled.setter
        def shadowsEnabled(self, arg: bool, /) -> None: ...

        @property
        def updateEveryFrame(self) -> bool: ...

        @updateEveryFrame.setter
        def updateEveryFrame(self, arg: bool, /) -> None: ...

        @property
        def contactHardening(self) -> bool: ...

        @contactHardening.setter
        def contactHardening(self, arg: bool, /) -> None: ...

        @property
        def cascadeRendered(self) -> bool: ...

        @cascadeRendered.setter
        def cascadeRendered(self, arg: bool, /) -> None: ...

        @property
        def atlasRendered(self) -> bool: ...

        @atlasRendered.setter
        def atlasRendered(self, arg: bool, /) -> None: ...

        @property
        def atlasOverflow(self) -> bool: ...

        @atlasOverflow.setter
        def atlasOverflow(self, arg: bool, /) -> None: ...

        @property
        def splitOverridesEnabled(self) -> bool: ...

        @splitOverridesEnabled.setter
        def splitOverridesEnabled(self, arg: bool, /) -> None: ...

        @property
        def cascadeCount(self) -> int: ...

        @cascadeCount.setter
        def cascadeCount(self, arg: int, /) -> None: ...

        @property
        def atlasTileCount(self) -> int: ...

        @atlasTileCount.setter
        def atlasTileCount(self, arg: int, /) -> None: ...

        @property
        def atlasNonPointTileCount(self) -> int: ...

        @atlasNonPointTileCount.setter
        def atlasNonPointTileCount(self, arg: int, /) -> None: ...

        @property
        def pointShadowFaceTileCount(self) -> int: ...

        @pointShadowFaceTileCount.setter
        def pointShadowFaceTileCount(self, arg: int, /) -> None: ...

        @property
        def selectedAdditionalShadowLights(self) -> int: ...

        @selectedAdditionalShadowLights.setter
        def selectedAdditionalShadowLights(self, arg: int, /) -> None: ...

        @property
        def droppedShadowCandidates(self) -> int: ...

        @droppedShadowCandidates.setter
        def droppedShadowCandidates(self, arg: int, /) -> None: ...

        @property
        def shadowBias(self) -> float: ...

        @shadowBias.setter
        def shadowBias(self, arg: float, /) -> None: ...

        @property
        def shadowNormalOffset(self) -> float: ...

        @shadowNormalOffset.setter
        def shadowNormalOffset(self, arg: float, /) -> None: ...

        @property
        def shadowPcfRadius(self) -> float: ...

        @shadowPcfRadius.setter
        def shadowPcfRadius(self, arg: float, /) -> None: ...

        @property
        def shadowStrength(self) -> float: ...

        @shadowStrength.setter
        def shadowStrength(self, arg: float, /) -> None: ...

        @property
        def cascadeLambda(self) -> float: ...

        @cascadeLambda.setter
        def cascadeLambda(self, arg: float, /) -> None: ...

        @property
        def cascadeMaxDistance(self) -> float: ...

        @cascadeMaxDistance.setter
        def cascadeMaxDistance(self, arg: float, /) -> None: ...

        @property
        def atlasOccupancy(self) -> float: ...

        @atlasOccupancy.setter
        def atlasOccupancy(self, arg: float, /) -> None: ...

        @property
        def cascadeSplits(self) -> list[float]: ...

        @cascadeSplits.setter
        def cascadeSplits(self, arg: Sequence[float], /) -> None: ...

        @property
        def cascadeSplitOverrides(self) -> list[float]: ...

        @cascadeSplitOverrides.setter
        def cascadeSplitOverrides(self, arg: Sequence[float], /) -> None: ...

        @property
        def cascadeMode(self) -> str: ...

        @cascadeMode.setter
        def cascadeMode(self, arg: str, /) -> None: ...

        @property
        def atlasMode(self) -> str: ...

        @atlasMode.setter
        def atlasMode(self, arg: str, /) -> None: ...

    class SupersampledCapture:
        def __init__(self) -> None: ...

        @property
        def width(self) -> int: ...

        @width.setter
        def width(self, arg: int, /) -> None: ...

        @property
        def height(self) -> int: ...

        @height.setter
        def height(self, arg: int, /) -> None: ...

        @property
        def scale(self) -> int: ...

        @scale.setter
        def scale(self, arg: int, /) -> None: ...

        @property
        def rgba(self) -> list[int]: ...

        @rgba.setter
        def rgba(self, arg: Sequence[int], /) -> None: ...

    class SynchronousMeshLoadingScope:
        def __init__(self, viewer: RayraiWindow, window: RayraiWindow) -> None: ...

        def __enter__(self) -> RayraiWindow.SynchronousMeshLoadingScope: ...

        def __exit__(self, exc_type: object | None, exc_value: object | None, traceback: object | None) -> None: ...

    class ThreadingMode(enum.Enum):
        SingleThread = 0

        MultiThread = 1

    class TransparentDrawDebugView:
        def __init__(self) -> None: ...

        @property
        def sortTransparentInstances(self) -> bool: ...

        @sortTransparentInstances.setter
        def sortTransparentInstances(self, arg: bool, /) -> None: ...

        @property
        def weightedBlendedOit(self) -> bool: ...

        @weightedBlendedOit.setter
        def weightedBlendedOit(self, arg: bool, /) -> None: ...

        @property
        def weightedBlendedOitAvailable(self) -> bool: ...

        @weightedBlendedOitAvailable.setter
        def weightedBlendedOitAvailable(self, arg: bool, /) -> None: ...

        @property
        def transparentDepthPrepass(self) -> bool: ...

        @transparentDepthPrepass.setter
        def transparentDepthPrepass(self, arg: bool, /) -> None: ...

        @property
        def transparentShadowDithering(self) -> bool: ...

        @transparentShadowDithering.setter
        def transparentShadowDithering(self, arg: bool, /) -> None: ...

        @property
        def transparentShadowOpacityScale(self) -> float: ...

        @transparentShadowOpacityScale.setter
        def transparentShadowOpacityScale(self, arg: float, /) -> None: ...

        @property
        def screenSpaceRefraction(self) -> bool: ...

        @screenSpaceRefraction.setter
        def screenSpaceRefraction(self, arg: bool, /) -> None: ...

        @property
        def screenSpaceRefractionAvailable(self) -> bool: ...

        @screenSpaceRefractionAvailable.setter
        def screenSpaceRefractionAvailable(self, arg: bool, /) -> None: ...

        @property
        def screenSpaceRefractionStrength(self) -> float: ...

        @screenSpaceRefractionStrength.setter
        def screenSpaceRefractionStrength(self, arg: float, /) -> None: ...

        @property
        def screenSpaceRefractionMaxPixels(self) -> float: ...

        @screenSpaceRefractionMaxPixels.setter
        def screenSpaceRefractionMaxPixels(self, arg: float, /) -> None: ...

        @property
        def transparentFogAlphaWeighted(self) -> bool: ...

        @transparentFogAlphaWeighted.setter
        def transparentFogAlphaWeighted(self, arg: bool, /) -> None: ...

        @property
        def transparentFogStrength(self) -> float: ...

        @transparentFogStrength.setter
        def transparentFogStrength(self, arg: float, /) -> None: ...

        @property
        def alphaToCoverage(self) -> bool: ...

        @alphaToCoverage.setter
        def alphaToCoverage(self, arg: bool, /) -> None: ...

        @property
        def depthWrite(self) -> bool: ...

        @depthWrite.setter
        def depthWrite(self, arg: bool, /) -> None: ...

        @property
        def depthTest(self) -> bool: ...

        @depthTest.setter
        def depthTest(self, arg: bool, /) -> None: ...

        @property
        def overdrawTileSize(self) -> int: ...

        @overdrawTileSize.setter
        def overdrawTileSize(self, arg: int, /) -> None: ...

        @property
        def overdrawTileColumns(self) -> int: ...

        @overdrawTileColumns.setter
        def overdrawTileColumns(self, arg: int, /) -> None: ...

        @property
        def overdrawTileRows(self) -> int: ...

        @overdrawTileRows.setter
        def overdrawTileRows(self, arg: int, /) -> None: ...

        @property
        def overdrawCoveredTileCount(self) -> int: ...

        @overdrawCoveredTileCount.setter
        def overdrawCoveredTileCount(self, arg: int, /) -> None: ...

        @property
        def overdrawHotTileCount(self) -> int: ...

        @overdrawHotTileCount.setter
        def overdrawHotTileCount(self, arg: int, /) -> None: ...

        @property
        def maxEstimatedOverdraw(self) -> int: ...

        @maxEstimatedOverdraw.setter
        def maxEstimatedOverdraw(self, arg: int, /) -> None: ...

        @property
        def averageEstimatedOverdraw(self) -> float: ...

        @averageEstimatedOverdraw.setter
        def averageEstimatedOverdraw(self, arg: float, /) -> None: ...

        @property
        def overdrawTileCounts(self) -> list[int]: ...

        @overdrawTileCounts.setter
        def overdrawTileCounts(self, arg: Sequence[int], /) -> None: ...

        @property
        def worldTransparentCount(self) -> int: ...

        @worldTransparentCount.setter
        def worldTransparentCount(self, arg: int, /) -> None: ...

        @property
        def visualTransparentCount(self) -> int: ...

        @visualTransparentCount.setter
        def visualTransparentCount(self, arg: int, /) -> None: ...

        @property
        def instancedTransparentCount(self) -> int: ...

        @instancedTransparentCount.setter
        def instancedTransparentCount(self, arg: int, /) -> None: ...

        @property
        def totalTransparentCount(self) -> int: ...

        @totalTransparentCount.setter
        def totalTransparentCount(self, arg: int, /) -> None: ...

        @property
        def backToFrontOrder(self) -> list[RayraiWindow.TransparentDrawItem]: ...

        @backToFrontOrder.setter
        def backToFrontOrder(self, arg: Sequence[RayraiWindow.TransparentDrawItem], /) -> None: ...

    class TransparentDrawItem:
        def __init__(self) -> None: ...

        @property
        def source(self) -> str: ...

        @source.setter
        def source(self, arg: str, /) -> None: ...

        @property
        def name(self) -> str: ...

        @name.setter
        def name(self, arg: str, /) -> None: ...

        @property
        def center(self) -> numpy.ndarray: ...

        @center.setter
        def center(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def distanceSquared(self) -> float: ...

        @distanceSquared.setter
        def distanceSquared(self, arg: float, /) -> None: ...

        @property
        def sortOffset(self) -> float: ...

        @sortOffset.setter
        def sortOffset(self, arg: float, /) -> None: ...

        @property
        def renderPriority(self) -> int: ...

        @renderPriority.setter
        def renderPriority(self, arg: int, /) -> None: ...

        @property
        def detectable(self) -> bool: ...

        @detectable.setter
        def detectable(self, arg: bool, /) -> None: ...

        @property
        def sortUsesBoundsCenter(self) -> bool: ...

        @sortUsesBoundsCenter.setter
        def sortUsesBoundsCenter(self, arg: bool, /) -> None: ...

    class VisualMeshBuilder:
        def __init__(self, viewer: RayraiWindow, renderer: RayraiWindow, name: str, meshFile: str) -> None: ...

        @overload
        def scale(self, s: numpy.ndarray) -> RayraiWindow.VisualMeshBuilder:
            """
            /// Per-axis mesh scale (default 1). @param s Scale. @return *this.
            C++: VisualMeshBuilder &(const glm::dvec3 &)
            """

        @overload
        def scale(self, x: float, y: float, z: float) -> RayraiWindow.VisualMeshBuilder:
            """
            /// Per-axis mesh scale. @param x X scale. @param y Y scale. @param z Z scale. @return *this.
            C++: VisualMeshBuilder &(double, double, double)
            """

        def color(self, c: numpy.ndarray) -> RayraiWindow.VisualMeshBuilder:
            """
            /// RGBA color in [0, 1] (default white). @param c Color. @return *this.
            C++: VisualMeshBuilder &(const glm::vec4 &)
            """

        def position(self, p: numpy.ndarray) -> RayraiWindow.VisualMeshBuilder:
            """
            /// World-space position. @param p Position in metres. @return *this.
            C++: VisualMeshBuilder &(const glm::vec3 &)
            """

        def orientation(self, q: numpy.ndarray) -> RayraiWindow.VisualMeshBuilder:
            """
            /// World-space orientation. @param q Unit quaternion. @return *this.
            C++: VisualMeshBuilder &(const glm::quat &)
            """

        def twoSided(self, enabled: bool = True) -> RayraiWindow.VisualMeshBuilder:
            """
            /// Disable back-face culling. @param enabled Flag. @return *this.
            C++: VisualMeshBuilder &(bool)
            """

        def pickable(self, enabled: bool = True) -> RayraiWindow.VisualMeshBuilder:
            """
            /// Sets Visuals::setDetectable(), which controls picking and visibility in
                /// sensor-style captures. @param enabled Flag. @return *this.
            C++: VisualMeshBuilder &(bool)
            """

        def castsShadows(self, enabled: bool = True) -> RayraiWindow.VisualMeshBuilder:
            """
            /// Whether the visual casts shadows. @param enabled Flag. @return *this.
            C++: VisualMeshBuilder &(bool)
            """

        def useMeshColor(self, enabled: bool) -> RayraiWindow.VisualMeshBuilder:
            """
            /// Use the mesh file's own colors instead of color(). @param enabled Flag. @return *this.
            C++: VisualMeshBuilder &(bool)
            """

        @overload
        def material(self, m: Material) -> RayraiWindow.VisualMeshBuilder:
            """
            /// Material override shared with the caller. @param m Material. @return *this.
            C++: VisualMeshBuilder &(std::shared_ptr<Material>)
            """

        @overload
        def material(self, m: Material) -> RayraiWindow.VisualMeshBuilder:
            """
            /// Material override (copied). @param m Material. @return *this.
            C++: VisualMeshBuilder &(const Material &)
            """

        def pbrEnvironment(self, env: PbrEnvironment) -> RayraiWindow.VisualMeshBuilder:
            """
            /// Image-based lighting maps for the visual. @param env Environment. @return *this.
            C++: VisualMeshBuilder &(const PbrEnvironment &)
            """

        def add(self) -> Visuals:
            """
            /** @brief Create the visual, apply the deferred state, return the shared_ptr.
                 *  @return The new visual (the name must be unique, `RSFATAL` otherwise). */
            C++: std::shared_ptr<Visuals> ()
            """

    class WeatherSidecarSettings:
        def __init__(self) -> None: ...

        @property
        def hasWeather(self) -> bool: ...

        @hasWeather.setter
        def hasWeather(self, arg: bool, /) -> None: ...

        @property
        def weather(self) -> WeatherSettings: ...

        @weather.setter
        def weather(self, arg: WeatherSettings, /) -> None: ...

        @property
        def hasLocalFogVolumes(self) -> bool: ...

        @hasLocalFogVolumes.setter
        def hasLocalFogVolumes(self, arg: bool, /) -> None: ...

        @property
        def localFogVolumes(self) -> list[LocalFogVolume]: ...

        @localFogVolumes.setter
        def localFogVolumes(self, arg: Sequence[LocalFogVolume], /) -> None: ...

    class AdditionalLight:
        def __init__(self) -> None: ...

        @property
        def type(self) -> LightType: ...

        @type.setter
        def type(self, arg: LightType, /) -> None: ...

        @property
        def position(self) -> numpy.ndarray: ...

        @position.setter
        def position(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def direction(self) -> numpy.ndarray: ...

        @direction.setter
        def direction(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def ambient(self) -> numpy.ndarray: ...

        @ambient.setter
        def ambient(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def diffuse(self) -> numpy.ndarray: ...

        @diffuse.setter
        def diffuse(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def specular(self) -> numpy.ndarray: ...

        @specular.setter
        def specular(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def constant(self) -> float: ...

        @constant.setter
        def constant(self, arg: float, /) -> None: ...

        @property
        def linear(self) -> float: ...

        @linear.setter
        def linear(self, arg: float, /) -> None: ...

        @property
        def quadratic(self) -> float: ...

        @quadratic.setter
        def quadratic(self, arg: float, /) -> None: ...

        @property
        def spotInnerCos(self) -> float: ...

        @spotInnerCos.setter
        def spotInnerCos(self, arg: float, /) -> None: ...

        @property
        def spotOuterCos(self) -> float: ...

        @spotOuterCos.setter
        def spotOuterCos(self, arg: float, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def areaSize(self) -> numpy.ndarray: ...

        @areaSize.setter
        def areaSize(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def areaRight(self) -> numpy.ndarray: ...

        @areaRight.setter
        def areaRight(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def areaUp(self) -> numpy.ndarray: ...

        @areaUp.setter
        def areaUp(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def projectorMap(self) -> int: ...

        @projectorMap.setter
        def projectorMap(self, arg: int, /) -> None: ...

        @property
        def projectorStrength(self) -> float: ...

        @projectorStrength.setter
        def projectorStrength(self, arg: float, /) -> None: ...

        @property
        def projectorUvScaleOffset(self) -> numpy.ndarray: ...

        @projectorUvScaleOffset.setter
        def projectorUvScaleOffset(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def negative(self) -> bool: ...

        @negative.setter
        def negative(self, arg: bool, /) -> None: ...

        @property
        def temperatureEnabled(self) -> bool: ...

        @temperatureEnabled.setter
        def temperatureEnabled(self, arg: bool, /) -> None: ...

        @property
        def temperatureKelvin(self) -> float: ...

        @temperatureKelvin.setter
        def temperatureKelvin(self, arg: float, /) -> None: ...

        @property
        def distanceFadeEnabled(self) -> bool: ...

        @distanceFadeEnabled.setter
        def distanceFadeEnabled(self, arg: bool, /) -> None: ...

        @property
        def distanceFadeBegin(self) -> float: ...

        @distanceFadeBegin.setter
        def distanceFadeBegin(self, arg: float, /) -> None: ...

        @property
        def distanceFadeShadow(self) -> float: ...

        @distanceFadeShadow.setter
        def distanceFadeShadow(self, arg: float, /) -> None: ...

        @property
        def distanceFadeLength(self) -> float: ...

        @distanceFadeLength.setter
        def distanceFadeLength(self, arg: float, /) -> None: ...

        @property
        def castsShadows(self) -> bool: ...

        @castsShadows.setter
        def castsShadows(self, arg: bool, /) -> None: ...

    class RenderQualityPreset(enum.Enum):
        Fast = 0

        Balanced = 1

        High = 2

        Ultra = 3

        Custom = 4

    class ViewerColorMode(enum.Enum):
        FastLinear = 0

        AcesApprox = 1

        UnrealPreviewApprox = 2

        FilmicApprox = 3

        AgXApprox = 4

    class ColorGradePreset(enum.Enum):
        Neutral = 0

        Warm = 1

        Cool = 2

        Cinematic = 3

        Bleach = 4

    class GlowBlendMode(enum.Enum):
        Additive = 0

        Screen = 1

        Softlight = 2

        Replace = 3

        Mix = 4

    class ViewerUpscaleFilter(enum.Enum):
        Linear = 0

        Mitchell = 1

    class PostProcessDebugMode(enum.Enum):
        Final = 0

        BloomSource = 1

        AmbientOcclusion = 2

        AmbientOcclusionRaw = 3

        FogTransmittance = 4

        FogDensity = 5

        FogScattering = 6

        WorldNormal = 7

        DepthHeatmap = 8

        ScreenSpaceIndirectLighting = 9

    class PointShadowPolicy(enum.Enum):
        Disabled = 0

        StandaloneCubemap = 1

    class RenderQualitySettings:
        def __init__(self) -> None: ...

        class BloomSettings:
            def __init__(self) -> None: ...

            @property
            def enabled(self) -> bool: ...

            @enabled.setter
            def enabled(self, arg: bool, /) -> None: ...

            @property
            def threshold(self) -> float: ...

            @threshold.setter
            def threshold(self, arg: float, /) -> None: ...

            @property
            def strength(self) -> float: ...

            @strength.setter
            def strength(self, arg: float, /) -> None: ...

            @property
            def radius(self) -> float: ...

            @radius.setter
            def radius(self, arg: float, /) -> None: ...

            @property
            def knee(self) -> float: ...

            @knee.setter
            def knee(self, arg: float, /) -> None: ...

            @property
            def quality(self) -> int: ...

            @quality.setter
            def quality(self, arg: int, /) -> None: ...

            @property
            def sourceClamp(self) -> float: ...

            @sourceClamp.setter
            def sourceClamp(self, arg: float, /) -> None: ...

            @property
            def anamorphic(self) -> float: ...

            @anamorphic.setter
            def anamorphic(self, arg: float, /) -> None: ...

            @property
            def dirtStrength(self) -> float: ...

            @dirtStrength.setter
            def dirtStrength(self, arg: float, /) -> None: ...

            @property
            def dirtScale(self) -> float: ...

            @dirtScale.setter
            def dirtScale(self, arg: float, /) -> None: ...

            @property
            def dirtTexture(self) -> int: ...

            @dirtTexture.setter
            def dirtTexture(self, arg: int, /) -> None: ...

            @property
            def blendMode(self) -> GlowBlendMode: ...

            @blendMode.setter
            def blendMode(self, arg: GlowBlendMode, /) -> None: ...

            @property
            def mix(self) -> float: ...

            @mix.setter
            def mix(self, arg: float, /) -> None: ...

        class DepthOfFieldSettings:
            def __init__(self) -> None: ...

            @property
            def enabled(self) -> bool: ...

            @enabled.setter
            def enabled(self, arg: bool, /) -> None: ...

            @property
            def focusDistance(self) -> float: ...

            @focusDistance.setter
            def focusDistance(self, arg: float, /) -> None: ...

            @property
            def focusRange(self) -> float: ...

            @focusRange.setter
            def focusRange(self, arg: float, /) -> None: ...

            @property
            def maxRadius(self) -> float: ...

            @maxRadius.setter
            def maxRadius(self, arg: float, /) -> None: ...

        class FogSettings:
            def __init__(self) -> None: ...

            @property
            def density(self) -> float: ...

            @density.setter
            def density(self, arg: float, /) -> None: ...

            @property
            def colorOverrideEnabled(self) -> bool: ...

            @colorOverrideEnabled.setter
            def colorOverrideEnabled(self, arg: bool, /) -> None: ...

            @property
            def color(self) -> numpy.ndarray: ...

            @color.setter
            def color(self, arg: numpy.ndarray, /) -> None: ...

            @property
            def heightEnabled(self) -> bool: ...

            @heightEnabled.setter
            def heightEnabled(self, arg: bool, /) -> None: ...

            @property
            def heightDensity(self) -> float: ...

            @heightDensity.setter
            def heightDensity(self, arg: float, /) -> None: ...

            @property
            def heightBaseHeight(self) -> float: ...

            @heightBaseHeight.setter
            def heightBaseHeight(self, arg: float, /) -> None: ...

            @property
            def heightFalloff(self) -> float: ...

            @heightFalloff.setter
            def heightFalloff(self, arg: float, /) -> None: ...

            @property
            def volumetricEnabled(self) -> bool: ...

            @volumetricEnabled.setter
            def volumetricEnabled(self, arg: bool, /) -> None: ...

            @property
            def volumetricDensity(self) -> float: ...

            @volumetricDensity.setter
            def volumetricDensity(self, arg: float, /) -> None: ...

            @property
            def volumetricNoiseScale(self) -> float: ...

            @volumetricNoiseScale.setter
            def volumetricNoiseScale(self, arg: float, /) -> None: ...

            @property
            def volumetricNoiseStrength(self) -> float: ...

            @volumetricNoiseStrength.setter
            def volumetricNoiseStrength(self, arg: float, /) -> None: ...

            @property
            def volumetricColor(self) -> numpy.ndarray: ...

            @volumetricColor.setter
            def volumetricColor(self, arg: numpy.ndarray, /) -> None: ...

            @property
            def volumetricAnisotropy(self) -> float: ...

            @volumetricAnisotropy.setter
            def volumetricAnisotropy(self, arg: float, /) -> None: ...

        class ShadowsSettings:
            def __init__(self) -> None: ...

            @property
            def enabled(self) -> bool: ...

            @enabled.setter
            def enabled(self, arg: bool, /) -> None: ...

            @property
            def resolution(self) -> int: ...

            @resolution.setter
            def resolution(self, arg: int, /) -> None: ...

            @property
            def bias(self) -> float: ...

            @bias.setter
            def bias(self, arg: float, /) -> None: ...

            @property
            def strength(self) -> float: ...

            @strength.setter
            def strength(self, arg: float, /) -> None: ...

            @property
            def pcfRadius(self) -> float: ...

            @pcfRadius.setter
            def pcfRadius(self, arg: float, /) -> None: ...

            @property
            def contactHardening(self) -> bool: ...

            @contactHardening.setter
            def contactHardening(self, arg: bool, /) -> None: ...

            @property
            def contactHardeningMinRadius(self) -> float: ...

            @contactHardeningMinRadius.setter
            def contactHardeningMinRadius(self, arg: float, /) -> None: ...

            @property
            def contactHardeningMaxRadius(self) -> float: ...

            @contactHardeningMaxRadius.setter
            def contactHardeningMaxRadius(self, arg: float, /) -> None: ...

            @property
            def contactHardeningDepthScale(self) -> float: ...

            @contactHardeningDepthScale.setter
            def contactHardeningDepthScale(self, arg: float, /) -> None: ...

            @property
            def orthoHalfSize(self) -> float: ...

            @orthoHalfSize.setter
            def orthoHalfSize(self, arg: float, /) -> None: ...

            @property
            def nearPlane(self) -> float: ...

            @nearPlane.setter
            def nearPlane(self, arg: float, /) -> None: ...

            @property
            def farPlane(self) -> float: ...

            @farPlane.setter
            def farPlane(self, arg: float, /) -> None: ...

            @property
            def centerOffset(self) -> float: ...

            @centerOffset.setter
            def centerOffset(self, arg: float, /) -> None: ...

            @property
            def directionalCascadeCount(self) -> int: ...

            @directionalCascadeCount.setter
            def directionalCascadeCount(self, arg: int, /) -> None: ...

            @property
            def directionalCascadeLambda(self) -> float: ...

            @directionalCascadeLambda.setter
            def directionalCascadeLambda(self, arg: float, /) -> None: ...

            @property
            def directionalCascadeMaxDistance(self) -> float: ...

            @directionalCascadeMaxDistance.setter
            def directionalCascadeMaxDistance(self, arg: float, /) -> None: ...

            @property
            def updateEveryFrame(self) -> bool: ...

            @updateEveryFrame.setter
            def updateEveryFrame(self, arg: bool, /) -> None: ...

            @property
            def directionalSoftnessDegrees(self) -> float: ...

            @directionalSoftnessDegrees.setter
            def directionalSoftnessDegrees(self, arg: float, /) -> None: ...

            @property
            def directionalSoftnessMaxRadius(self) -> float: ...

            @directionalSoftnessMaxRadius.setter
            def directionalSoftnessMaxRadius(self, arg: float, /) -> None: ...

        class TemporalAaSettings:
            def __init__(self) -> None: ...

            @property
            def enabled(self) -> bool: ...

            @enabled.setter
            def enabled(self, arg: bool, /) -> None: ...

            @property
            def blend(self) -> float: ...

            @blend.setter
            def blend(self, arg: float, /) -> None: ...

            @property
            def jitterScale(self) -> float: ...

            @jitterScale.setter
            def jitterScale(self, arg: float, /) -> None: ...

            @property
            def method(self) -> TemporalAaMethod: ...

            @method.setter
            def method(self, arg: TemporalAaMethod, /) -> None: ...

            @property
            def currentWeight(self) -> float: ...

            @currentWeight.setter
            def currentWeight(self, arg: float, /) -> None: ...

            @property
            def sharpness(self) -> float: ...

            @sharpness.setter
            def sharpness(self, arg: float, /) -> None: ...

        class ToneMappingSettings:
            def __init__(self) -> None: ...

            @property
            def pbrToneMappingEnabled(self) -> bool: ...

            @pbrToneMappingEnabled.setter
            def pbrToneMappingEnabled(self, arg: bool, /) -> None: ...

            @property
            def exposure(self) -> float: ...

            @exposure.setter
            def exposure(self, arg: float, /) -> None: ...

            @property
            def autoExposureEnabled(self) -> bool: ...

            @autoExposureEnabled.setter
            def autoExposureEnabled(self, arg: bool, /) -> None: ...

            @property
            def autoExposureKey(self) -> float: ...

            @autoExposureKey.setter
            def autoExposureKey(self, arg: float, /) -> None: ...

            @property
            def autoExposureSpeed(self) -> float: ...

            @autoExposureSpeed.setter
            def autoExposureSpeed(self, arg: float, /) -> None: ...

            @property
            def autoExposureMinFactor(self) -> float: ...

            @autoExposureMinFactor.setter
            def autoExposureMinFactor(self, arg: float, /) -> None: ...

            @property
            def autoExposureMaxFactor(self) -> float: ...

            @autoExposureMaxFactor.setter
            def autoExposureMaxFactor(self, arg: float, /) -> None: ...

            @property
            def whiteBalance(self) -> numpy.ndarray: ...

            @whiteBalance.setter
            def whiteBalance(self, arg: numpy.ndarray, /) -> None: ...

            @property
            def saturation(self) -> float: ...

            @saturation.setter
            def saturation(self, arg: float, /) -> None: ...

            @property
            def gradePreset(self) -> ColorGradePreset: ...

            @gradePreset.setter
            def gradePreset(self, arg: ColorGradePreset, /) -> None: ...

            @property
            def gradeStrength(self) -> float: ...

            @gradeStrength.setter
            def gradeStrength(self, arg: float, /) -> None: ...

        @property
        def backgroundColorRgb255(self) -> numpy.ndarray: ...

        @backgroundColorRgb255.setter
        def backgroundColorRgb255(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def mainLightAmbient(self) -> numpy.ndarray: ...

        @mainLightAmbient.setter
        def mainLightAmbient(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def mainLightDiffuse(self) -> numpy.ndarray: ...

        @mainLightDiffuse.setter
        def mainLightDiffuse(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def mainLightSpecular(self) -> numpy.ndarray: ...

        @mainLightSpecular.setter
        def mainLightSpecular(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def mainLightDirection(self) -> numpy.ndarray: ...

        @mainLightDirection.setter
        def mainLightDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def shadowResolution(self) -> int: ...

        @shadowResolution.setter
        def shadowResolution(self, arg: int, /) -> None: ...

        @property
        def shadowBias(self) -> float: ...

        @shadowBias.setter
        def shadowBias(self, arg: float, /) -> None: ...

        @property
        def shadowStrength(self) -> float: ...

        @shadowStrength.setter
        def shadowStrength(self, arg: float, /) -> None: ...

        @property
        def shadowPcfRadius(self) -> float: ...

        @shadowPcfRadius.setter
        def shadowPcfRadius(self, arg: float, /) -> None: ...

        @property
        def directionalShadowSoftnessDegrees(self) -> float: ...

        @directionalShadowSoftnessDegrees.setter
        def directionalShadowSoftnessDegrees(self, arg: float, /) -> None: ...

        @property
        def directionalShadowSoftnessMaxRadius(self) -> float: ...

        @directionalShadowSoftnessMaxRadius.setter
        def directionalShadowSoftnessMaxRadius(self, arg: float, /) -> None: ...

        @property
        def shadowContactHardening(self) -> bool: ...

        @shadowContactHardening.setter
        def shadowContactHardening(self, arg: bool, /) -> None: ...

        @property
        def shadowContactHardeningMinRadius(self) -> float: ...

        @shadowContactHardeningMinRadius.setter
        def shadowContactHardeningMinRadius(self, arg: float, /) -> None: ...

        @property
        def shadowContactHardeningMaxRadius(self) -> float: ...

        @shadowContactHardeningMaxRadius.setter
        def shadowContactHardeningMaxRadius(self, arg: float, /) -> None: ...

        @property
        def shadowContactHardeningDepthScale(self) -> float: ...

        @shadowContactHardeningDepthScale.setter
        def shadowContactHardeningDepthScale(self, arg: float, /) -> None: ...

        @property
        def shadowDebugOverlayEnabled(self) -> bool: ...

        @shadowDebugOverlayEnabled.setter
        def shadowDebugOverlayEnabled(self, arg: bool, /) -> None: ...

        @property
        def shadowOrthoHalfSize(self) -> float: ...

        @shadowOrthoHalfSize.setter
        def shadowOrthoHalfSize(self, arg: float, /) -> None: ...

        @property
        def shadowNear(self) -> float: ...

        @shadowNear.setter
        def shadowNear(self, arg: float, /) -> None: ...

        @property
        def shadowFar(self) -> float: ...

        @shadowFar.setter
        def shadowFar(self, arg: float, /) -> None: ...

        @property
        def shadowCenterOffset(self) -> float: ...

        @shadowCenterOffset.setter
        def shadowCenterOffset(self, arg: float, /) -> None: ...

        @property
        def fogDensity(self) -> float: ...

        @fogDensity.setter
        def fogDensity(self, arg: float, /) -> None: ...

        @property
        def fogColorOverrideEnabled(self) -> bool: ...

        @fogColorOverrideEnabled.setter
        def fogColorOverrideEnabled(self, arg: bool, /) -> None: ...

        @property
        def fogColor(self) -> numpy.ndarray: ...

        @fogColor.setter
        def fogColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def heightFogEnabled(self) -> bool: ...

        @heightFogEnabled.setter
        def heightFogEnabled(self, arg: bool, /) -> None: ...

        @property
        def heightFogDensity(self) -> float: ...

        @heightFogDensity.setter
        def heightFogDensity(self, arg: float, /) -> None: ...

        @property
        def heightFogBaseHeight(self) -> float: ...

        @heightFogBaseHeight.setter
        def heightFogBaseHeight(self, arg: float, /) -> None: ...

        @property
        def heightFogFalloff(self) -> float: ...

        @heightFogFalloff.setter
        def heightFogFalloff(self, arg: float, /) -> None: ...

        @property
        def volumetricFogEnabled(self) -> bool: ...

        @volumetricFogEnabled.setter
        def volumetricFogEnabled(self, arg: bool, /) -> None: ...

        @property
        def volumetricFogDensity(self) -> float: ...

        @volumetricFogDensity.setter
        def volumetricFogDensity(self, arg: float, /) -> None: ...

        @property
        def volumetricFogNoiseScale(self) -> float: ...

        @volumetricFogNoiseScale.setter
        def volumetricFogNoiseScale(self, arg: float, /) -> None: ...

        @property
        def volumetricFogNoiseStrength(self) -> float: ...

        @volumetricFogNoiseStrength.setter
        def volumetricFogNoiseStrength(self, arg: float, /) -> None: ...

        @property
        def volumetricFogColor(self) -> numpy.ndarray: ...

        @volumetricFogColor.setter
        def volumetricFogColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def volumetricFogAnisotropy(self) -> float: ...

        @volumetricFogAnisotropy.setter
        def volumetricFogAnisotropy(self, arg: float, /) -> None: ...

        @property
        def volumetricFogAnimationTimeSeconds(self) -> float: ...

        @volumetricFogAnimationTimeSeconds.setter
        def volumetricFogAnimationTimeSeconds(self, arg: float, /) -> None: ...

        @property
        def volumetricFogWindDirection(self) -> numpy.ndarray: ...

        @volumetricFogWindDirection.setter
        def volumetricFogWindDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def volumetricFogWindSpeed(self) -> float: ...

        @volumetricFogWindSpeed.setter
        def volumetricFogWindSpeed(self, arg: float, /) -> None: ...

        @property
        def volumetricFogTurbulenceSpeed(self) -> float: ...

        @volumetricFogTurbulenceSpeed.setter
        def volumetricFogTurbulenceSpeed(self, arg: float, /) -> None: ...

        @property
        def volumetricLightingEnabled(self) -> bool: ...

        @volumetricLightingEnabled.setter
        def volumetricLightingEnabled(self, arg: bool, /) -> None: ...

        @property
        def volumetricLightStrength(self) -> float: ...

        @volumetricLightStrength.setter
        def volumetricLightStrength(self, arg: float, /) -> None: ...

        @property
        def volumetricLightDecay(self) -> float: ...

        @volumetricLightDecay.setter
        def volumetricLightDecay(self, arg: float, /) -> None: ...

        @property
        def volumetricLightSamples(self) -> int: ...

        @volumetricLightSamples.setter
        def volumetricLightSamples(self, arg: int, /) -> None: ...

        @property
        def gamma(self) -> float: ...

        @gamma.setter
        def gamma(self, arg: float, /) -> None: ...

        @property
        def highFidelityPbr(self) -> bool: ...

        @highFidelityPbr.setter
        def highFidelityPbr(self, arg: bool, /) -> None: ...

        @property
        def forceSimpleMaterialShading(self) -> bool: ...

        @forceSimpleMaterialShading.setter
        def forceSimpleMaterialShading(self, arg: bool, /) -> None: ...

        @property
        def pbrToneMapping(self) -> bool: ...

        @pbrToneMapping.setter
        def pbrToneMapping(self, arg: bool, /) -> None: ...

        @property
        def colorMode(self) -> ViewerColorMode: ...

        @colorMode.setter
        def colorMode(self, arg: ViewerColorMode, /) -> None: ...

        @property
        def pbrExposure(self) -> float: ...

        @pbrExposure.setter
        def pbrExposure(self, arg: float, /) -> None: ...

        @property
        def autoExposureEnabled(self) -> bool: ...

        @autoExposureEnabled.setter
        def autoExposureEnabled(self, arg: bool, /) -> None: ...

        @property
        def autoExposureKey(self) -> float: ...

        @autoExposureKey.setter
        def autoExposureKey(self, arg: float, /) -> None: ...

        @property
        def autoExposureSpeed(self) -> float: ...

        @autoExposureSpeed.setter
        def autoExposureSpeed(self, arg: float, /) -> None: ...

        @property
        def autoExposureMinFactor(self) -> float: ...

        @autoExposureMinFactor.setter
        def autoExposureMinFactor(self, arg: float, /) -> None: ...

        @property
        def autoExposureMaxFactor(self) -> float: ...

        @autoExposureMaxFactor.setter
        def autoExposureMaxFactor(self, arg: float, /) -> None: ...

        @property
        def viewerWhiteBalance(self) -> numpy.ndarray: ...

        @viewerWhiteBalance.setter
        def viewerWhiteBalance(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def viewerSaturation(self) -> float: ...

        @viewerSaturation.setter
        def viewerSaturation(self, arg: float, /) -> None: ...

        @property
        def viewerColorGradePreset(self) -> ColorGradePreset: ...

        @viewerColorGradePreset.setter
        def viewerColorGradePreset(self, arg: ColorGradePreset, /) -> None: ...

        @property
        def viewerColorGradeStrength(self) -> float: ...

        @viewerColorGradeStrength.setter
        def viewerColorGradeStrength(self, arg: float, /) -> None: ...

        @property
        def pbrEnvironmentMaxLod(self) -> float: ...

        @pbrEnvironmentMaxLod.setter
        def pbrEnvironmentMaxLod(self, arg: float, /) -> None: ...

        @property
        def pbrEnvironmentMipLodBias(self) -> float: ...

        @pbrEnvironmentMipLodBias.setter
        def pbrEnvironmentMipLodBias(self, arg: float, /) -> None: ...

        @property
        def pbrEnvironmentRotation(self) -> float: ...

        @pbrEnvironmentRotation.setter
        def pbrEnvironmentRotation(self, arg: float, /) -> None: ...

        @property
        def pbrEnvironmentIntensity(self) -> float: ...

        @pbrEnvironmentIntensity.setter
        def pbrEnvironmentIntensity(self, arg: float, /) -> None: ...

        @property
        def pbrEnvironmentLightingTint(self) -> numpy.ndarray: ...

        @pbrEnvironmentLightingTint.setter
        def pbrEnvironmentLightingTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def pbrEnvironmentSkyTint(self) -> numpy.ndarray: ...

        @pbrEnvironmentSkyTint.setter
        def pbrEnvironmentSkyTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def pbrEnvironmentHorizonTint(self) -> numpy.ndarray: ...

        @pbrEnvironmentHorizonTint.setter
        def pbrEnvironmentHorizonTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def pbrEnvironmentGroundTint(self) -> numpy.ndarray: ...

        @pbrEnvironmentGroundTint.setter
        def pbrEnvironmentGroundTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def pbrEnvironmentHorizonStrength(self) -> float: ...

        @pbrEnvironmentHorizonStrength.setter
        def pbrEnvironmentHorizonStrength(self, arg: float, /) -> None: ...

        @property
        def environmentBackgroundTint(self) -> numpy.ndarray: ...

        @environmentBackgroundTint.setter
        def environmentBackgroundTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def environmentBackgroundExposureScale(self) -> float: ...

        @environmentBackgroundExposureScale.setter
        def environmentBackgroundExposureScale(self, arg: float, /) -> None: ...

        @property
        def proceduralSkyBackgroundEnabled(self) -> bool: ...

        @proceduralSkyBackgroundEnabled.setter
        def proceduralSkyBackgroundEnabled(self, arg: bool, /) -> None: ...

        @property
        def proceduralSkySunStrength(self) -> float: ...

        @proceduralSkySunStrength.setter
        def proceduralSkySunStrength(self, arg: float, /) -> None: ...

        @property
        def proceduralSkySunSize(self) -> float: ...

        @proceduralSkySunSize.setter
        def proceduralSkySunSize(self, arg: float, /) -> None: ...

        @property
        def proceduralCloudLayerEnabled(self) -> bool: ...

        @proceduralCloudLayerEnabled.setter
        def proceduralCloudLayerEnabled(self, arg: bool, /) -> None: ...

        @property
        def proceduralCloudCoverage(self) -> float: ...

        @proceduralCloudCoverage.setter
        def proceduralCloudCoverage(self, arg: float, /) -> None: ...

        @property
        def proceduralCloudDensity(self) -> float: ...

        @proceduralCloudDensity.setter
        def proceduralCloudDensity(self, arg: float, /) -> None: ...

        @property
        def proceduralCloudScale(self) -> float: ...

        @proceduralCloudScale.setter
        def proceduralCloudScale(self, arg: float, /) -> None: ...

        @property
        def proceduralCloudSoftness(self) -> float: ...

        @proceduralCloudSoftness.setter
        def proceduralCloudSoftness(self, arg: float, /) -> None: ...

        @property
        def proceduralCloudOffset(self) -> numpy.ndarray: ...

        @proceduralCloudOffset.setter
        def proceduralCloudOffset(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def proceduralCloudTint(self) -> numpy.ndarray: ...

        @proceduralCloudTint.setter
        def proceduralCloudTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def cloudQuality(self) -> CloudQuality: ...

        @cloudQuality.setter
        def cloudQuality(self, arg: CloudQuality, /) -> None: ...

        @property
        def cloudShadowProjectionEnabled(self) -> bool: ...

        @cloudShadowProjectionEnabled.setter
        def cloudShadowProjectionEnabled(self, arg: bool, /) -> None: ...

        @property
        def cloudShadowStrength(self) -> float: ...

        @cloudShadowStrength.setter
        def cloudShadowStrength(self, arg: float, /) -> None: ...

        @property
        def cloudShadowScale(self) -> float: ...

        @cloudShadowScale.setter
        def cloudShadowScale(self, arg: float, /) -> None: ...

        @property
        def pbrKeyLightIntensity(self) -> float: ...

        @pbrKeyLightIntensity.setter
        def pbrKeyLightIntensity(self, arg: float, /) -> None: ...

        @property
        def specularAntiAliasing(self) -> bool: ...

        @specularAntiAliasing.setter
        def specularAntiAliasing(self, arg: bool, /) -> None: ...

        @property
        def specularAntiAliasingStrength(self) -> float: ...

        @specularAntiAliasingStrength.setter
        def specularAntiAliasingStrength(self, arg: float, /) -> None: ...

        @property
        def reflectiveGround(self) -> bool: ...

        @reflectiveGround.setter
        def reflectiveGround(self, arg: bool, /) -> None: ...

        @property
        def reflectiveGroundRoughness(self) -> float: ...

        @reflectiveGroundRoughness.setter
        def reflectiveGroundRoughness(self, arg: float, /) -> None: ...

        @property
        def reflectiveGroundMetallic(self) -> float: ...

        @reflectiveGroundMetallic.setter
        def reflectiveGroundMetallic(self, arg: float, /) -> None: ...

        @property
        def fxaaEnabled(self) -> bool: ...

        @fxaaEnabled.setter
        def fxaaEnabled(self, arg: bool, /) -> None: ...

        @property
        def viewerVignetteStrength(self) -> float: ...

        @viewerVignetteStrength.setter
        def viewerVignetteStrength(self, arg: float, /) -> None: ...

        @property
        def viewerChromaticAberrationStrength(self) -> float: ...

        @viewerChromaticAberrationStrength.setter
        def viewerChromaticAberrationStrength(self, arg: float, /) -> None: ...

        @property
        def viewerFilmGrainStrength(self) -> float: ...

        @viewerFilmGrainStrength.setter
        def viewerFilmGrainStrength(self, arg: float, /) -> None: ...

        @property
        def aerialPerspectiveEnabled(self) -> bool: ...

        @aerialPerspectiveEnabled.setter
        def aerialPerspectiveEnabled(self, arg: bool, /) -> None: ...

        @property
        def aerialPerspectiveStrength(self) -> float: ...

        @aerialPerspectiveStrength.setter
        def aerialPerspectiveStrength(self, arg: float, /) -> None: ...

        @property
        def detailNormalEnabled(self) -> bool: ...

        @detailNormalEnabled.setter
        def detailNormalEnabled(self, arg: bool, /) -> None: ...

        @property
        def detailNormalScale(self) -> float: ...

        @detailNormalScale.setter
        def detailNormalScale(self, arg: float, /) -> None: ...

        @property
        def detailNormalStrength(self) -> float: ...

        @detailNormalStrength.setter
        def detailNormalStrength(self, arg: float, /) -> None: ...

        @property
        def viewerSubsurfaceWrap(self) -> float: ...

        @viewerSubsurfaceWrap.setter
        def viewerSubsurfaceWrap(self, arg: float, /) -> None: ...

        @property
        def viewerSubsurfaceTint(self) -> numpy.ndarray: ...

        @viewerSubsurfaceTint.setter
        def viewerSubsurfaceTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lightShaftsEnabled(self) -> bool: ...

        @lightShaftsEnabled.setter
        def lightShaftsEnabled(self, arg: bool, /) -> None: ...

        @property
        def lightShaftsStrength(self) -> float: ...

        @lightShaftsStrength.setter
        def lightShaftsStrength(self, arg: float, /) -> None: ...

        @property
        def lightShaftsDecay(self) -> float: ...

        @lightShaftsDecay.setter
        def lightShaftsDecay(self, arg: float, /) -> None: ...

        @property
        def lightShaftsDensity(self) -> float: ...

        @lightShaftsDensity.setter
        def lightShaftsDensity(self, arg: float, /) -> None: ...

        @property
        def lensFlareGhostStrength(self) -> float: ...

        @lensFlareGhostStrength.setter
        def lensFlareGhostStrength(self, arg: float, /) -> None: ...

        @property
        def lensFlareStreakStrength(self) -> float: ...

        @lensFlareStreakStrength.setter
        def lensFlareStreakStrength(self, arg: float, /) -> None: ...

        @property
        def lensFlareTint(self) -> numpy.ndarray: ...

        @lensFlareTint.setter
        def lensFlareTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def contactShadowsEnabled(self) -> bool: ...

        @contactShadowsEnabled.setter
        def contactShadowsEnabled(self, arg: bool, /) -> None: ...

        @property
        def contactShadowsLength(self) -> float: ...

        @contactShadowsLength.setter
        def contactShadowsLength(self, arg: float, /) -> None: ...

        @property
        def contactShadowsStrength(self) -> float: ...

        @contactShadowsStrength.setter
        def contactShadowsStrength(self, arg: float, /) -> None: ...

        @property
        def contactShadowsThickness(self) -> float: ...

        @contactShadowsThickness.setter
        def contactShadowsThickness(self, arg: float, /) -> None: ...

        @property
        def motionBlurEnabled(self) -> bool: ...

        @motionBlurEnabled.setter
        def motionBlurEnabled(self, arg: bool, /) -> None: ...

        @property
        def motionBlurDirection(self) -> numpy.ndarray: ...

        @motionBlurDirection.setter
        def motionBlurDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def motionBlurStrength(self) -> float: ...

        @motionBlurStrength.setter
        def motionBlurStrength(self, arg: float, /) -> None: ...

        @property
        def ssrEnabled(self) -> bool: ...

        @ssrEnabled.setter
        def ssrEnabled(self, arg: bool, /) -> None: ...

        @property
        def ssrStrength(self) -> float: ...

        @ssrStrength.setter
        def ssrStrength(self, arg: float, /) -> None: ...

        @property
        def ssrSteps(self) -> int: ...

        @ssrSteps.setter
        def ssrSteps(self, arg: int, /) -> None: ...

        @property
        def ssrMaxDistance(self) -> float: ...

        @ssrMaxDistance.setter
        def ssrMaxDistance(self, arg: float, /) -> None: ...

        @property
        def ssrThickness(self) -> float: ...

        @ssrThickness.setter
        def ssrThickness(self, arg: float, /) -> None: ...

        @property
        def lensDistortionEnabled(self) -> bool: ...

        @lensDistortionEnabled.setter
        def lensDistortionEnabled(self, arg: bool, /) -> None: ...

        @property
        def lensDistortionStrength(self) -> float: ...

        @lensDistortionStrength.setter
        def lensDistortionStrength(self, arg: float, /) -> None: ...

        @property
        def lensDistortionCenter(self) -> numpy.ndarray: ...

        @lensDistortionCenter.setter
        def lensDistortionCenter(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def pixelateEnabled(self) -> bool: ...

        @pixelateEnabled.setter
        def pixelateEnabled(self, arg: bool, /) -> None: ...

        @property
        def pixelateBlockSize(self) -> int: ...

        @pixelateBlockSize.setter
        def pixelateBlockSize(self, arg: int, /) -> None: ...

        @property
        def nightVisionEnabled(self) -> bool: ...

        @nightVisionEnabled.setter
        def nightVisionEnabled(self, arg: bool, /) -> None: ...

        @property
        def nightVisionStrength(self) -> float: ...

        @nightVisionStrength.setter
        def nightVisionStrength(self, arg: float, /) -> None: ...

        @property
        def nightVisionGain(self) -> float: ...

        @nightVisionGain.setter
        def nightVisionGain(self, arg: float, /) -> None: ...

        @property
        def ditherEnabled(self) -> bool: ...

        @ditherEnabled.setter
        def ditherEnabled(self, arg: bool, /) -> None: ...

        @property
        def ditherStrength(self) -> float: ...

        @ditherStrength.setter
        def ditherStrength(self, arg: float, /) -> None: ...

        @property
        def thermalEnabled(self) -> bool: ...

        @thermalEnabled.setter
        def thermalEnabled(self, arg: bool, /) -> None: ...

        @property
        def thermalStrength(self) -> float: ...

        @thermalStrength.setter
        def thermalStrength(self, arg: float, /) -> None: ...

        @property
        def thermalGain(self) -> float: ...

        @thermalGain.setter
        def thermalGain(self, arg: float, /) -> None: ...

        @property
        def scanlineEnabled(self) -> bool: ...

        @scanlineEnabled.setter
        def scanlineEnabled(self, arg: bool, /) -> None: ...

        @property
        def scanlineStrength(self) -> float: ...

        @scanlineStrength.setter
        def scanlineStrength(self, arg: float, /) -> None: ...

        @property
        def scanlineFrequency(self) -> float: ...

        @scanlineFrequency.setter
        def scanlineFrequency(self, arg: float, /) -> None: ...

        @property
        def letterboxEnabled(self) -> bool: ...

        @letterboxEnabled.setter
        def letterboxEnabled(self, arg: bool, /) -> None: ...

        @property
        def letterboxAspect(self) -> float: ...

        @letterboxAspect.setter
        def letterboxAspect(self, arg: float, /) -> None: ...

        @property
        def letterboxColor(self) -> numpy.ndarray: ...

        @letterboxColor.setter
        def letterboxColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def underwaterEnabled(self) -> bool: ...

        @underwaterEnabled.setter
        def underwaterEnabled(self, arg: bool, /) -> None: ...

        @property
        def underwaterTint(self) -> numpy.ndarray: ...

        @underwaterTint.setter
        def underwaterTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def underwaterCausticStrength(self) -> float: ...

        @underwaterCausticStrength.setter
        def underwaterCausticStrength(self, arg: float, /) -> None: ...

        @property
        def underwaterCausticScale(self) -> float: ...

        @underwaterCausticScale.setter
        def underwaterCausticScale(self, arg: float, /) -> None: ...

        @property
        def underwaterDepthExtinction(self) -> float: ...

        @underwaterDepthExtinction.setter
        def underwaterDepthExtinction(self, arg: float, /) -> None: ...

        @property
        def heatHazeEnabled(self) -> bool: ...

        @heatHazeEnabled.setter
        def heatHazeEnabled(self, arg: bool, /) -> None: ...

        @property
        def heatHazeStrength(self) -> float: ...

        @heatHazeStrength.setter
        def heatHazeStrength(self, arg: float, /) -> None: ...

        @property
        def heatHazeFrequency(self) -> float: ...

        @heatHazeFrequency.setter
        def heatHazeFrequency(self, arg: float, /) -> None: ...

        @property
        def heatHazeSpeed(self) -> float: ...

        @heatHazeSpeed.setter
        def heatHazeSpeed(self, arg: float, /) -> None: ...

        @property
        def heatHazeMaxY(self) -> float: ...

        @heatHazeMaxY.setter
        def heatHazeMaxY(self, arg: float, /) -> None: ...

        @property
        def zoomBlurEnabled(self) -> bool: ...

        @zoomBlurEnabled.setter
        def zoomBlurEnabled(self, arg: bool, /) -> None: ...

        @property
        def zoomBlurStrength(self) -> float: ...

        @zoomBlurStrength.setter
        def zoomBlurStrength(self, arg: float, /) -> None: ...

        @property
        def zoomBlurCenter(self) -> numpy.ndarray: ...

        @zoomBlurCenter.setter
        def zoomBlurCenter(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def zoomBlurInnerRadius(self) -> float: ...

        @zoomBlurInnerRadius.setter
        def zoomBlurInnerRadius(self, arg: float, /) -> None: ...

        @property
        def posterizeEnabled(self) -> bool: ...

        @posterizeEnabled.setter
        def posterizeEnabled(self, arg: bool, /) -> None: ...

        @property
        def posterizeLevels(self) -> int: ...

        @posterizeLevels.setter
        def posterizeLevels(self, arg: int, /) -> None: ...

        @property
        def starburstEnabled(self) -> bool: ...

        @starburstEnabled.setter
        def starburstEnabled(self, arg: bool, /) -> None: ...

        @property
        def starburstStrength(self) -> float: ...

        @starburstStrength.setter
        def starburstStrength(self, arg: float, /) -> None: ...

        @property
        def starburstSpikes(self) -> int: ...

        @starburstSpikes.setter
        def starburstSpikes(self, arg: int, /) -> None: ...

        @property
        def starburstAngleOffset(self) -> float: ...

        @starburstAngleOffset.setter
        def starburstAngleOffset(self, arg: float, /) -> None: ...

        @property
        def starburstRadius(self) -> float: ...

        @starburstRadius.setter
        def starburstRadius(self, arg: float, /) -> None: ...

        @property
        def starburstTint(self) -> numpy.ndarray: ...

        @starburstTint.setter
        def starburstTint(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def bloomEnabled(self) -> bool: ...

        @bloomEnabled.setter
        def bloomEnabled(self, arg: bool, /) -> None: ...

        @property
        def bloomThreshold(self) -> float: ...

        @bloomThreshold.setter
        def bloomThreshold(self, arg: float, /) -> None: ...

        @property
        def bloomStrength(self) -> float: ...

        @bloomStrength.setter
        def bloomStrength(self, arg: float, /) -> None: ...

        @property
        def bloomRadius(self) -> float: ...

        @bloomRadius.setter
        def bloomRadius(self, arg: float, /) -> None: ...

        @property
        def bloomKnee(self) -> float: ...

        @bloomKnee.setter
        def bloomKnee(self, arg: float, /) -> None: ...

        @property
        def bloomQuality(self) -> int: ...

        @bloomQuality.setter
        def bloomQuality(self, arg: int, /) -> None: ...

        @property
        def bloomSourceClamp(self) -> float: ...

        @bloomSourceClamp.setter
        def bloomSourceClamp(self, arg: float, /) -> None: ...

        @property
        def bloomAnamorphic(self) -> float: ...

        @bloomAnamorphic.setter
        def bloomAnamorphic(self, arg: float, /) -> None: ...

        @property
        def bloomDirtStrength(self) -> float: ...

        @bloomDirtStrength.setter
        def bloomDirtStrength(self, arg: float, /) -> None: ...

        @property
        def bloomDirtScale(self) -> float: ...

        @bloomDirtScale.setter
        def bloomDirtScale(self, arg: float, /) -> None: ...

        @property
        def bloomDirtTexture(self) -> int: ...

        @bloomDirtTexture.setter
        def bloomDirtTexture(self, arg: int, /) -> None: ...

        @property
        def bloomBlendMode(self) -> GlowBlendMode: ...

        @bloomBlendMode.setter
        def bloomBlendMode(self, arg: GlowBlendMode, /) -> None: ...

        @property
        def bloomMix(self) -> float: ...

        @bloomMix.setter
        def bloomMix(self, arg: float, /) -> None: ...

        @property
        def postProcessDebugMode(self) -> PostProcessDebugMode: ...

        @postProcessDebugMode.setter
        def postProcessDebugMode(self, arg: PostProcessDebugMode, /) -> None: ...

        @property
        def screenSpaceAoEnabled(self) -> bool: ...

        @screenSpaceAoEnabled.setter
        def screenSpaceAoEnabled(self, arg: bool, /) -> None: ...

        @property
        def screenSpaceAoRadius(self) -> float: ...

        @screenSpaceAoRadius.setter
        def screenSpaceAoRadius(self, arg: float, /) -> None: ...

        @property
        def screenSpaceAoStrength(self) -> float: ...

        @screenSpaceAoStrength.setter
        def screenSpaceAoStrength(self, arg: float, /) -> None: ...

        @property
        def screenSpaceAoBias(self) -> float: ...

        @screenSpaceAoBias.setter
        def screenSpaceAoBias(self, arg: float, /) -> None: ...

        @property
        def screenSpaceAoSamples(self) -> int: ...

        @screenSpaceAoSamples.setter
        def screenSpaceAoSamples(self, arg: int, /) -> None: ...

        @property
        def screenSpaceAoFalloff(self) -> float: ...

        @screenSpaceAoFalloff.setter
        def screenSpaceAoFalloff(self, arg: float, /) -> None: ...

        @property
        def screenSpaceAoRadiusWorldSpace(self) -> bool: ...

        @screenSpaceAoRadiusWorldSpace.setter
        def screenSpaceAoRadiusWorldSpace(self, arg: bool, /) -> None: ...

        @property
        def screenSpaceAoMaxPixelRadius(self) -> float: ...

        @screenSpaceAoMaxPixelRadius.setter
        def screenSpaceAoMaxPixelRadius(self, arg: float, /) -> None: ...

        @property
        def screenSpaceAoGeometryAware(self) -> bool: ...

        @screenSpaceAoGeometryAware.setter
        def screenSpaceAoGeometryAware(self, arg: bool, /) -> None: ...

        @property
        def contactAoRadius(self) -> float: ...

        @contactAoRadius.setter
        def contactAoRadius(self, arg: float, /) -> None: ...

        @property
        def contactAoStrength(self) -> float: ...

        @contactAoStrength.setter
        def contactAoStrength(self, arg: float, /) -> None: ...

        @property
        def contactAoSamples(self) -> int: ...

        @contactAoSamples.setter
        def contactAoSamples(self, arg: int, /) -> None: ...

        @property
        def contactAoFalloff(self) -> float: ...

        @contactAoFalloff.setter
        def contactAoFalloff(self, arg: float, /) -> None: ...

        @property
        def screenSpaceAoDenoiseEnabled(self) -> bool: ...

        @screenSpaceAoDenoiseEnabled.setter
        def screenSpaceAoDenoiseEnabled(self, arg: bool, /) -> None: ...

        @property
        def screenSpaceAoDenoiseStrength(self) -> float: ...

        @screenSpaceAoDenoiseStrength.setter
        def screenSpaceAoDenoiseStrength(self, arg: float, /) -> None: ...

        @property
        def screenSpaceAoDenoiseRadius(self) -> int: ...

        @screenSpaceAoDenoiseRadius.setter
        def screenSpaceAoDenoiseRadius(self, arg: int, /) -> None: ...

        @property
        def screenSpaceAoDenoiseDepthSigma(self) -> float: ...

        @screenSpaceAoDenoiseDepthSigma.setter
        def screenSpaceAoDenoiseDepthSigma(self, arg: float, /) -> None: ...

        @property
        def screenSpaceIndirectLightingEnabled(self) -> bool: ...

        @screenSpaceIndirectLightingEnabled.setter
        def screenSpaceIndirectLightingEnabled(self, arg: bool, /) -> None: ...

        @property
        def screenSpaceIndirectLightingRadius(self) -> float: ...

        @screenSpaceIndirectLightingRadius.setter
        def screenSpaceIndirectLightingRadius(self, arg: float, /) -> None: ...

        @property
        def screenSpaceIndirectLightingStrength(self) -> float: ...

        @screenSpaceIndirectLightingStrength.setter
        def screenSpaceIndirectLightingStrength(self, arg: float, /) -> None: ...

        @property
        def screenSpaceIndirectLightingSamples(self) -> int: ...

        @screenSpaceIndirectLightingSamples.setter
        def screenSpaceIndirectLightingSamples(self, arg: int, /) -> None: ...

        @property
        def screenSpaceIndirectLightingFalloff(self) -> float: ...

        @screenSpaceIndirectLightingFalloff.setter
        def screenSpaceIndirectLightingFalloff(self, arg: float, /) -> None: ...

        @property
        def screenSpaceIndirectLightingNormalRejection(self) -> float: ...

        @screenSpaceIndirectLightingNormalRejection.setter
        def screenSpaceIndirectLightingNormalRejection(self, arg: float, /) -> None: ...

        @property
        def screenSpaceIndirectLightingSaturation(self) -> float: ...

        @screenSpaceIndirectLightingSaturation.setter
        def screenSpaceIndirectLightingSaturation(self, arg: float, /) -> None: ...

        @property
        def opaqueDepthPrepass(self) -> bool: ...

        @opaqueDepthPrepass.setter
        def opaqueDepthPrepass(self, arg: bool, /) -> None: ...

        @property
        def frontToBackOpaqueInstancedOrdering(self) -> bool: ...

        @frontToBackOpaqueInstancedOrdering.setter
        def frontToBackOpaqueInstancedOrdering(self, arg: bool, /) -> None: ...

        @property
        def foliageAlphaDepthPrepass(self) -> bool: ...

        @foliageAlphaDepthPrepass.setter
        def foliageAlphaDepthPrepass(self, arg: bool, /) -> None: ...

        @property
        def depthOfFieldEnabled(self) -> bool: ...

        @depthOfFieldEnabled.setter
        def depthOfFieldEnabled(self, arg: bool, /) -> None: ...

        @property
        def depthOfFieldFocusDistance(self) -> float: ...

        @depthOfFieldFocusDistance.setter
        def depthOfFieldFocusDistance(self, arg: float, /) -> None: ...

        @property
        def depthOfFieldFocusRange(self) -> float: ...

        @depthOfFieldFocusRange.setter
        def depthOfFieldFocusRange(self, arg: float, /) -> None: ...

        @property
        def depthOfFieldMaxRadius(self) -> float: ...

        @depthOfFieldMaxRadius.setter
        def depthOfFieldMaxRadius(self, arg: float, /) -> None: ...

        @property
        def textureAnisotropy(self) -> float: ...

        @textureAnisotropy.setter
        def textureAnisotropy(self, arg: float, /) -> None: ...

        @property
        def textureMipLodBias(self) -> float: ...

        @textureMipLodBias.setter
        def textureMipLodBias(self, arg: float, /) -> None: ...

        @property
        def viewerMsaaSamples(self) -> int: ...

        @viewerMsaaSamples.setter
        def viewerMsaaSamples(self, arg: int, /) -> None: ...

        @property
        def alphaToCoverage(self) -> bool: ...

        @alphaToCoverage.setter
        def alphaToCoverage(self, arg: bool, /) -> None: ...

        @property
        def temporalAaEnabled(self) -> bool: ...

        @temporalAaEnabled.setter
        def temporalAaEnabled(self, arg: bool, /) -> None: ...

        @property
        def temporalAaBlend(self) -> float: ...

        @temporalAaBlend.setter
        def temporalAaBlend(self, arg: float, /) -> None: ...

        @property
        def temporalAaJitterScale(self) -> float: ...

        @temporalAaJitterScale.setter
        def temporalAaJitterScale(self, arg: float, /) -> None: ...

        @property
        def temporalAaMethod(self) -> TemporalAaMethod: ...

        @temporalAaMethod.setter
        def temporalAaMethod(self, arg: TemporalAaMethod, /) -> None: ...

        @property
        def temporalAaCurrentWeight(self) -> float: ...

        @temporalAaCurrentWeight.setter
        def temporalAaCurrentWeight(self, arg: float, /) -> None: ...

        @property
        def temporalAaSharpness(self) -> float: ...

        @temporalAaSharpness.setter
        def temporalAaSharpness(self, arg: float, /) -> None: ...

        @property
        def viewerRenderScale(self) -> float: ...

        @viewerRenderScale.setter
        def viewerRenderScale(self, arg: float, /) -> None: ...

        @property
        def viewerUpscaleFilter(self) -> ViewerUpscaleFilter: ...

        @viewerUpscaleFilter.setter
        def viewerUpscaleFilter(self, arg: ViewerUpscaleFilter, /) -> None: ...

        @property
        def shadowsEnabled(self) -> bool: ...

        @shadowsEnabled.setter
        def shadowsEnabled(self, arg: bool, /) -> None: ...

        @property
        def shadowedLightBudget(self) -> int: ...

        @shadowedLightBudget.setter
        def shadowedLightBudget(self, arg: int, /) -> None: ...

        @property
        def pointShadowPolicy(self) -> PointShadowPolicy: ...

        @pointShadowPolicy.setter
        def pointShadowPolicy(self, arg: PointShadowPolicy, /) -> None: ...

        @property
        def maxPointShadowLights(self) -> int: ...

        @maxPointShadowLights.setter
        def maxPointShadowLights(self, arg: int, /) -> None: ...

        @property
        def additionalShadowResolutionScale(self) -> float: ...

        @additionalShadowResolutionScale.setter
        def additionalShadowResolutionScale(self, arg: float, /) -> None: ...

        @property
        def pointShadowResolutionScale(self) -> float: ...

        @pointShadowResolutionScale.setter
        def pointShadowResolutionScale(self, arg: float, /) -> None: ...

        @property
        def minAdditionalShadowResolution(self) -> int: ...

        @minAdditionalShadowResolution.setter
        def minAdditionalShadowResolution(self, arg: int, /) -> None: ...

        @property
        def shadowSlopeBiasScale(self) -> float: ...

        @shadowSlopeBiasScale.setter
        def shadowSlopeBiasScale(self, arg: float, /) -> None: ...

        @property
        def shadowNormalOffset(self) -> float: ...

        @shadowNormalOffset.setter
        def shadowNormalOffset(self, arg: float, /) -> None: ...

        @property
        def directionalShadowCascadeCount(self) -> int: ...

        @directionalShadowCascadeCount.setter
        def directionalShadowCascadeCount(self, arg: int, /) -> None: ...

        @property
        def directionalShadowCascadeLambda(self) -> float: ...

        @directionalShadowCascadeLambda.setter
        def directionalShadowCascadeLambda(self, arg: float, /) -> None: ...

        @property
        def directionalShadowCascadeMaxDistance(self) -> float: ...

        @directionalShadowCascadeMaxDistance.setter
        def directionalShadowCascadeMaxDistance(self, arg: float, /) -> None: ...

        @property
        def directionalShadowCascadeSplitOverrides(self) -> list[float]: ...

        @directionalShadowCascadeSplitOverrides.setter
        def directionalShadowCascadeSplitOverrides(self, arg: Sequence[float], /) -> None: ...

        @property
        def additionalShadowBiasScale(self) -> float: ...

        @additionalShadowBiasScale.setter
        def additionalShadowBiasScale(self, arg: float, /) -> None: ...

        @property
        def additionalShadowStrengthScale(self) -> float: ...

        @additionalShadowStrengthScale.setter
        def additionalShadowStrengthScale(self, arg: float, /) -> None: ...

        @property
        def additionalShadowPcfRadiusScale(self) -> float: ...

        @additionalShadowPcfRadiusScale.setter
        def additionalShadowPcfRadiusScale(self, arg: float, /) -> None: ...

        @property
        def additionalShadowNormalOffsetScale(self) -> float: ...

        @additionalShadowNormalOffsetScale.setter
        def additionalShadowNormalOffsetScale(self, arg: float, /) -> None: ...

        @property
        def additionalShadowAtlasPlanningEnabled(self) -> bool: ...

        @additionalShadowAtlasPlanningEnabled.setter
        def additionalShadowAtlasPlanningEnabled(self, arg: bool, /) -> None: ...

        @property
        def additionalShadowAtlasResolution(self) -> int: ...

        @additionalShadowAtlasResolution.setter
        def additionalShadowAtlasResolution(self, arg: int, /) -> None: ...

        @property
        def additionalShadowAtlasPadding(self) -> int: ...

        @additionalShadowAtlasPadding.setter
        def additionalShadowAtlasPadding(self, arg: int, /) -> None: ...

        @property
        def pointShadowBiasScale(self) -> float: ...

        @pointShadowBiasScale.setter
        def pointShadowBiasScale(self, arg: float, /) -> None: ...

        @property
        def pointShadowFar(self) -> float: ...

        @pointShadowFar.setter
        def pointShadowFar(self, arg: float, /) -> None: ...

        @property
        def updateShadowsEveryFrame(self) -> bool: ...

        @updateShadowsEveryFrame.setter
        def updateShadowsEveryFrame(self, arg: bool, /) -> None: ...

        @property
        def maxAdditionalLightsPerFrame(self) -> int: ...

        @maxAdditionalLightsPerFrame.setter
        def maxAdditionalLightsPerFrame(self, arg: int, /) -> None: ...

        @property
        def minAdditionalLightInfluence(self) -> float: ...

        @minAdditionalLightInfluence.setter
        def minAdditionalLightInfluence(self, arg: float, /) -> None: ...

        @property
        def additionalLightContributionThreshold(self) -> float: ...

        @additionalLightContributionThreshold.setter
        def additionalLightContributionThreshold(self, arg: float, /) -> None: ...

        @property
        def maxReflectionProbesPerSelection(self) -> int: ...

        @maxReflectionProbesPerSelection.setter
        def maxReflectionProbesPerSelection(self, arg: int, /) -> None: ...

        @property
        def minReflectionProbeWeight(self) -> float: ...

        @minReflectionProbeWeight.setter
        def minReflectionProbeWeight(self, arg: float, /) -> None: ...

        @property
        def reflectionProbeDebugOverlayEnabled(self) -> bool: ...

        @reflectionProbeDebugOverlayEnabled.setter
        def reflectionProbeDebugOverlayEnabled(self, arg: bool, /) -> None: ...

        @property
        def reflectionProbeFilteringEnabled(self) -> bool: ...

        @reflectionProbeFilteringEnabled.setter
        def reflectionProbeFilteringEnabled(self, arg: bool, /) -> None: ...

        @property
        def reflectionProbeIrradianceResolution(self) -> int: ...

        @reflectionProbeIrradianceResolution.setter
        def reflectionProbeIrradianceResolution(self, arg: int, /) -> None: ...

        @property
        def reflectionProbeIrradianceSamples(self) -> int: ...

        @reflectionProbeIrradianceSamples.setter
        def reflectionProbeIrradianceSamples(self, arg: int, /) -> None: ...

        @property
        def reflectionProbePrefilteredResolution(self) -> int: ...

        @reflectionProbePrefilteredResolution.setter
        def reflectionProbePrefilteredResolution(self, arg: int, /) -> None: ...

        @property
        def reflectionProbePrefilteredMipLevels(self) -> int: ...

        @reflectionProbePrefilteredMipLevels.setter
        def reflectionProbePrefilteredMipLevels(self, arg: int, /) -> None: ...

        @property
        def reflectionProbePrefilteredSamples(self) -> int: ...

        @reflectionProbePrefilteredSamples.setter
        def reflectionProbePrefilteredSamples(self, arg: int, /) -> None: ...

        @property
        def reflectionProbeBrdfLutSize(self) -> int: ...

        @reflectionProbeBrdfLutSize.setter
        def reflectionProbeBrdfLutSize(self, arg: int, /) -> None: ...

        @property
        def reflectionProbeBrdfLutSamples(self) -> int: ...

        @reflectionProbeBrdfLutSamples.setter
        def reflectionProbeBrdfLutSamples(self, arg: int, /) -> None: ...

        @property
        def localDiffuseIrradianceIntensity(self) -> float: ...

        @localDiffuseIrradianceIntensity.setter
        def localDiffuseIrradianceIntensity(self, arg: float, /) -> None: ...

        @property
        def localDiffuseIrradianceSaturation(self) -> float: ...

        @localDiffuseIrradianceSaturation.setter
        def localDiffuseIrradianceSaturation(self, arg: float, /) -> None: ...

        @property
        def autoSelectImportedShadowLight(self) -> bool: ...

        @autoSelectImportedShadowLight.setter
        def autoSelectImportedShadowLight(self, arg: bool, /) -> None: ...

        @property
        def sortTransparentInstances(self) -> bool: ...

        @sortTransparentInstances.setter
        def sortTransparentInstances(self, arg: bool, /) -> None: ...

        @property
        def weightedBlendedOit(self) -> bool: ...

        @weightedBlendedOit.setter
        def weightedBlendedOit(self, arg: bool, /) -> None: ...

        @property
        def transparentDepthPrepass(self) -> bool: ...

        @transparentDepthPrepass.setter
        def transparentDepthPrepass(self, arg: bool, /) -> None: ...

        @property
        def transparentShadowDithering(self) -> bool: ...

        @transparentShadowDithering.setter
        def transparentShadowDithering(self, arg: bool, /) -> None: ...

        @property
        def transparentShadowOpacityScale(self) -> float: ...

        @transparentShadowOpacityScale.setter
        def transparentShadowOpacityScale(self, arg: float, /) -> None: ...

        @property
        def screenSpaceRefraction(self) -> bool: ...

        @screenSpaceRefraction.setter
        def screenSpaceRefraction(self, arg: bool, /) -> None: ...

        @property
        def screenSpaceRefractionStrength(self) -> float: ...

        @screenSpaceRefractionStrength.setter
        def screenSpaceRefractionStrength(self, arg: float, /) -> None: ...

        @property
        def screenSpaceRefractionMaxPixels(self) -> float: ...

        @screenSpaceRefractionMaxPixels.setter
        def screenSpaceRefractionMaxPixels(self, arg: float, /) -> None: ...

        @property
        def geometryRefraction(self) -> bool: ...

        @geometryRefraction.setter
        def geometryRefraction(self, arg: bool, /) -> None: ...

        @property
        def geometryRefractionBackend(self) -> GeometryRefractionBackend: ...

        @geometryRefractionBackend.setter
        def geometryRefractionBackend(self, arg: GeometryRefractionBackend, /) -> None: ...

        @property
        def geometryRefractionSamples(self) -> int: ...

        @geometryRefractionSamples.setter
        def geometryRefractionSamples(self, arg: int, /) -> None: ...

        @property
        def geometryRefractionMaxBounces(self) -> int: ...

        @geometryRefractionMaxBounces.setter
        def geometryRefractionMaxBounces(self, arg: int, /) -> None: ...

        @property
        def geometryRefractionProgressive(self) -> bool: ...

        @geometryRefractionProgressive.setter
        def geometryRefractionProgressive(self, arg: bool, /) -> None: ...

        @property
        def geometryRefractionMaxSamples(self) -> int: ...

        @geometryRefractionMaxSamples.setter
        def geometryRefractionMaxSamples(self, arg: int, /) -> None: ...

        @property
        def geometryRefractionLowDiscrepancy(self) -> bool: ...

        @geometryRefractionLowDiscrepancy.setter
        def geometryRefractionLowDiscrepancy(self, arg: bool, /) -> None: ...

        @property
        def geometryRefractionAdaptive(self) -> bool: ...

        @geometryRefractionAdaptive.setter
        def geometryRefractionAdaptive(self, arg: bool, /) -> None: ...

        @property
        def geometryRefractionMinSamples(self) -> int: ...

        @geometryRefractionMinSamples.setter
        def geometryRefractionMinSamples(self, arg: int, /) -> None: ...

        @property
        def geometryRefractionRelativeError(self) -> float: ...

        @geometryRefractionRelativeError.setter
        def geometryRefractionRelativeError(self, arg: float, /) -> None: ...

        @property
        def geometryRefractionAbsoluteError(self) -> float: ...

        @geometryRefractionAbsoluteError.setter
        def geometryRefractionAbsoluteError(self, arg: float, /) -> None: ...

        @property
        def geometryRefractionSeed(self) -> int: ...

        @geometryRefractionSeed.setter
        def geometryRefractionSeed(self, arg: int, /) -> None: ...

        @property
        def geometryRefractionRevision(self) -> int: ...

        @geometryRefractionRevision.setter
        def geometryRefractionRevision(self, arg: int, /) -> None: ...

        @property
        def transparentFogAlphaWeighted(self) -> bool: ...

        @transparentFogAlphaWeighted.setter
        def transparentFogAlphaWeighted(self, arg: bool, /) -> None: ...

        @property
        def transparentFogStrength(self) -> float: ...

        @transparentFogStrength.setter
        def transparentFogStrength(self, arg: float, /) -> None: ...

        @property
        def waterEnabled(self) -> bool: ...

        @waterEnabled.setter
        def waterEnabled(self, arg: bool, /) -> None: ...

        @property
        def waterTileResolution(self) -> int: ...

        @waterTileResolution.setter
        def waterTileResolution(self, arg: int, /) -> None: ...

        @property
        def waterLodDistanceScale(self) -> float: ...

        @waterLodDistanceScale.setter
        def waterLodDistanceScale(self, arg: float, /) -> None: ...

        @property
        def waterScreenSpaceReflections(self) -> bool: ...

        @waterScreenSpaceReflections.setter
        def waterScreenSpaceReflections(self, arg: bool, /) -> None: ...

        @property
        def waterReflectionSteps(self) -> int: ...

        @waterReflectionSteps.setter
        def waterReflectionSteps(self, arg: int, /) -> None: ...

        @property
        def waterReflectionMaxDistance(self) -> float: ...

        @waterReflectionMaxDistance.setter
        def waterReflectionMaxDistance(self, arg: float, /) -> None: ...

        @property
        def waterCaustics(self) -> bool: ...

        @waterCaustics.setter
        def waterCaustics(self, arg: bool, /) -> None: ...

        @property
        def waterUnderwaterView(self) -> bool: ...

        @waterUnderwaterView.setter
        def waterUnderwaterView(self, arg: bool, /) -> None: ...

        @property
        def waterSkyCaptureResolution(self) -> int: ...

        @waterSkyCaptureResolution.setter
        def waterSkyCaptureResolution(self, arg: int, /) -> None: ...

        @property
        def weatherWetMaterialEnabled(self) -> bool: ...

        @weatherWetMaterialEnabled.setter
        def weatherWetMaterialEnabled(self, arg: bool, /) -> None: ...

        @property
        def weatherWetness(self) -> float: ...

        @weatherWetness.setter
        def weatherWetness(self, arg: float, /) -> None: ...

        @property
        def weatherPuddleStrength(self) -> float: ...

        @weatherPuddleStrength.setter
        def weatherPuddleStrength(self, arg: float, /) -> None: ...

        @property
        def weatherRainRippleStrength(self) -> float: ...

        @weatherRainRippleStrength.setter
        def weatherRainRippleStrength(self, arg: float, /) -> None: ...

        @property
        def weatherRainRippleScale(self) -> float: ...

        @weatherRainRippleScale.setter
        def weatherRainRippleScale(self, arg: float, /) -> None: ...

        @property
        def weatherRainRipplePhase(self) -> float: ...

        @weatherRainRipplePhase.setter
        def weatherRainRipplePhase(self, arg: float, /) -> None: ...

        @property
        def weatherWetAlbedoDarkening(self) -> float: ...

        @weatherWetAlbedoDarkening.setter
        def weatherWetAlbedoDarkening(self, arg: float, /) -> None: ...

        @property
        def weatherWetRoughnessScale(self) -> float: ...

        @weatherWetRoughnessScale.setter
        def weatherWetRoughnessScale(self, arg: float, /) -> None: ...

        @property
        def weatherWetSpecularBoost(self) -> float: ...

        @weatherWetSpecularBoost.setter
        def weatherWetSpecularBoost(self, arg: float, /) -> None: ...

        @property
        def weatherSnowMaterialEnabled(self) -> bool: ...

        @weatherSnowMaterialEnabled.setter
        def weatherSnowMaterialEnabled(self, arg: bool, /) -> None: ...

        @property
        def weatherSnowCoverage(self) -> float: ...

        @weatherSnowCoverage.setter
        def weatherSnowCoverage(self, arg: float, /) -> None: ...

        @property
        def weatherSnowAccumulationStrength(self) -> float: ...

        @weatherSnowAccumulationStrength.setter
        def weatherSnowAccumulationStrength(self, arg: float, /) -> None: ...

        @property
        def weatherSnowAlbedoBlend(self) -> float: ...

        @weatherSnowAlbedoBlend.setter
        def weatherSnowAlbedoBlend(self, arg: float, /) -> None: ...

        @property
        def weatherSnowRoughness(self) -> float: ...

        @weatherSnowRoughness.setter
        def weatherSnowRoughness(self, arg: float, /) -> None: ...

        @property
        def weatherSnowMetallicScale(self) -> float: ...

        @weatherSnowMetallicScale.setter
        def weatherSnowMetallicScale(self, arg: float, /) -> None: ...

        @property
        def weatherSnowNormalSoftening(self) -> float: ...

        @weatherSnowNormalSoftening.setter
        def weatherSnowNormalSoftening(self, arg: float, /) -> None: ...

        @property
        def foliageWindEnabled(self) -> bool: ...

        @foliageWindEnabled.setter
        def foliageWindEnabled(self, arg: bool, /) -> None: ...

        @property
        def foliageWindDirection(self) -> numpy.ndarray: ...

        @foliageWindDirection.setter
        def foliageWindDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def foliageWindSpeed(self) -> float: ...

        @foliageWindSpeed.setter
        def foliageWindSpeed(self, arg: float, /) -> None: ...

        @property
        def foliageWindTimeSeconds(self) -> float: ...

        @foliageWindTimeSeconds.setter
        def foliageWindTimeSeconds(self, arg: float, /) -> None: ...

        @property
        def foliageWindGustStrength(self) -> float: ...

        @foliageWindGustStrength.setter
        def foliageWindGustStrength(self, arg: float, /) -> None: ...

        @property
        def foliageWindGustScale(self) -> float: ...

        @foliageWindGustScale.setter
        def foliageWindGustScale(self, arg: float, /) -> None: ...

        @property
        def foliageWindBranchBend(self) -> float: ...

        @foliageWindBranchBend.setter
        def foliageWindBranchBend(self, arg: float, /) -> None: ...

        @property
        def foliageWindLeafFlutter(self) -> float: ...

        @foliageWindLeafFlutter.setter
        def foliageWindLeafFlutter(self, arg: float, /) -> None: ...

        @property
        def addViewerFillLights(self) -> bool: ...

        @addViewerFillLights.setter
        def addViewerFillLights(self, arg: bool, /) -> None: ...

        def shadows(self) -> object:
            """
            /// @brief Read a snapshot of the shadows sub-config from the flat fields.
              /// @return Copy of the main shadow and cascade fields (additional-light shadow fields excluded).
            C++: ShadowsSettings () const
            """

        def setShadows(self, s: RenderQualitySettings.ShadowsSettings) -> object:
            """
            /// @brief Write the shadows sub-config back into the flat fields.
              /// @param s Values to store.
            C++: void (const ShadowsSettings &)
            """

        def bloom(self) -> object:
            """
            /// @brief Read a snapshot of the bloom sub-config from the flat fields.
              /// @return Copy of the bloom* fields.
            C++: BloomSettings () const
            """

        def setBloom(self, b: RenderQualitySettings.BloomSettings) -> object:
            """
            /// @brief Write the bloom sub-config back into the flat fields.
              /// @param b Values to store.
            C++: void (const BloomSettings &)
            """

        def depthOfField(self) -> object:
            """
            /// @brief Read a snapshot of the depth-of-field sub-config from the flat fields.
              /// @return Copy of the depthOfField* fields.
            C++: DepthOfFieldSettings () const
            """

        def setDepthOfField(self, d: RenderQualitySettings.DepthOfFieldSettings) -> object:
            """
            /// @brief Write the depth-of-field sub-config back into the flat fields.
              /// @param d Values to store.
            C++: void (const DepthOfFieldSettings &)
            """

        def fog(self) -> object:
            """
            /// @brief Read a snapshot of the fog sub-config from the flat fields.
              /// @return Copy of the distance/height/volumetric fog fields (wind and animation excluded).
            C++: FogSettings () const
            """

        def setFog(self, f: RenderQualitySettings.FogSettings) -> object:
            """
            /// @brief Write the fog sub-config back into the flat fields.
              /// @param f Values to store.
            C++: void (const FogSettings &)
            """

        def toneMapping(self) -> object:
            """
            /// @brief Read a snapshot of the tone-mapping and color sub-config from the flat fields.
              /// @return Copy of the tone-mapping, auto-exposure and color-grade fields (colorMode excluded).
            C++: ToneMappingSettings () const
            """

        def setToneMapping(self, t: RenderQualitySettings.ToneMappingSettings) -> object:
            """
            /// @brief Write the tone-mapping and color sub-config back into the flat fields.
              /// @param t Values to store.
            C++: void (const ToneMappingSettings &)
            """

        def temporalAa(self) -> object:
            """
            /// @brief Read a snapshot of the temporal-AA sub-config from the flat fields.
              /// @return Copy of the temporalAa* fields.
            C++: TemporalAaSettings () const
            """

        def setTemporalAa(self, a: RenderQualitySettings.TemporalAaSettings) -> object:
            """
            /// @brief Write the temporal-AA sub-config back into the flat fields.
              /// @param a Values to store.
            C++: void (const TemporalAaSettings &)
            """

        @staticmethod
        def preset(p: RenderQualityPreset) -> object:
            """
            /**
               * @brief Build a `RenderQualitySettings` for the given preset.
               *
               * Companion to the existing `RenderQualityPreset` enum so the type that
               * names the presets can also produce them. Delegates to
               * `RayraiWindow::defaultRenderQualitySettings()` (defined in
               * RenderQualityDefaults.cpp). The result is not clamped until it is passed to
               * `RayraiWindow::setRenderQualitySettings()`.
               *
               * @param p Preset to build.
               * @return Struct defaults with the preset's overrides applied.
               */
            C++: RenderQualitySettings (RenderQualityPreset)
            """

    class WeatherPreset(enum.Enum):
        Clear = 0

        Hazy = 1

        Overcast = 2

        Fog = 3

        Rain = 4

        HeavyRain = 5

        Snow = 6

        Storm = 7

        NightClear = 8

        NightRain = 9

        Custom = 10

    class WeatherQuality(enum.Enum):
        Low = 0

        Medium = 1

        High = 2

        Ultra = 3

    class LocalFogVolume:
        def __init__(self) -> None: ...

        @property
        def center(self) -> numpy.ndarray: ...

        @center.setter
        def center(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def color(self) -> numpy.ndarray: ...

        @color.setter
        def color(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def density(self) -> float: ...

        @density.setter
        def density(self, arg: float, /) -> None: ...

        @property
        def edgeFade(self) -> float: ...

        @edgeFade.setter
        def edgeFade(self, arg: float, /) -> None: ...

        @property
        def noiseScale(self) -> float: ...

        @noiseScale.setter
        def noiseScale(self, arg: float, /) -> None: ...

        @property
        def noiseStrength(self) -> float: ...

        @noiseStrength.setter
        def noiseStrength(self, arg: float, /) -> None: ...

    class WeatherSettings:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def preset(self) -> WeatherPreset: ...

        @preset.setter
        def preset(self, arg: WeatherPreset, /) -> None: ...

        @property
        def quality(self) -> WeatherQuality: ...

        @quality.setter
        def quality(self, arg: WeatherQuality, /) -> None: ...

        @property
        def seed(self) -> int: ...

        @seed.setter
        def seed(self, arg: int, /) -> None: ...

        @property
        def timeOfDayHours(self) -> float: ...

        @timeOfDayHours.setter
        def timeOfDayHours(self, arg: float, /) -> None: ...

        @property
        def latitude(self) -> float: ...

        @latitude.setter
        def latitude(self, arg: float, /) -> None: ...

        @property
        def longitude(self) -> float: ...

        @longitude.setter
        def longitude(self, arg: float, /) -> None: ...

        @property
        def automaticUtcOffset(self) -> bool: ...

        @automaticUtcOffset.setter
        def automaticUtcOffset(self, arg: bool, /) -> None: ...

        @property
        def utcOffsetHours(self) -> float: ...

        @utcOffsetHours.setter
        def utcOffsetHours(self, arg: float, /) -> None: ...

        @property
        def year(self) -> int: ...

        @year.setter
        def year(self, arg: int, /) -> None: ...

        @property
        def month(self) -> int: ...

        @month.setter
        def month(self, arg: int, /) -> None: ...

        @property
        def day(self) -> int: ...

        @day.setter
        def day(self, arg: int, /) -> None: ...

        @property
        def windDirection(self) -> numpy.ndarray: ...

        @windDirection.setter
        def windDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def windSpeed(self) -> float: ...

        @windSpeed.setter
        def windSpeed(self, arg: float, /) -> None: ...

        @property
        def transitionSeconds(self) -> float: ...

        @transitionSeconds.setter
        def transitionSeconds(self, arg: float, /) -> None: ...

        @property
        def affectSensors(self) -> bool: ...

        @affectSensors.setter
        def affectSensors(self, arg: bool, /) -> None: ...

        @property
        def cloudCoverage(self) -> float: ...

        @cloudCoverage.setter
        def cloudCoverage(self, arg: float, /) -> None: ...

        @property
        def cloudDensity(self) -> float: ...

        @cloudDensity.setter
        def cloudDensity(self, arg: float, /) -> None: ...

        @property
        def precipitationRate(self) -> float: ...

        @precipitationRate.setter
        def precipitationRate(self, arg: float, /) -> None: ...

        @property
        def rainOcclusionStrength(self) -> float: ...

        @rainOcclusionStrength.setter
        def rainOcclusionStrength(self, arg: float, /) -> None: ...

        @property
        def fogDensity(self) -> float: ...

        @fogDensity.setter
        def fogDensity(self, arg: float, /) -> None: ...

        @property
        def visibilityMeters(self) -> float: ...

        @visibilityMeters.setter
        def visibilityMeters(self, arg: float, /) -> None: ...

        @property
        def fogColor(self) -> numpy.ndarray: ...

        @fogColor.setter
        def fogColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def fogAnisotropy(self) -> float: ...

        @fogAnisotropy.setter
        def fogAnisotropy(self, arg: float, /) -> None: ...

        @property
        def humidity(self) -> float: ...

        @humidity.setter
        def humidity(self, arg: float, /) -> None: ...

        @property
        def wetness(self) -> float: ...

        @wetness.setter
        def wetness(self, arg: float, /) -> None: ...

        @property
        def wetnessAccumulationEnabled(self) -> bool: ...

        @wetnessAccumulationEnabled.setter
        def wetnessAccumulationEnabled(self, arg: bool, /) -> None: ...

        @property
        def wetnessAccumulationRate(self) -> float: ...

        @wetnessAccumulationRate.setter
        def wetnessAccumulationRate(self, arg: float, /) -> None: ...

        @property
        def wetnessDryingRate(self) -> float: ...

        @wetnessDryingRate.setter
        def wetnessDryingRate(self, arg: float, /) -> None: ...

        @property
        def snowCoverage(self) -> float: ...

        @snowCoverage.setter
        def snowCoverage(self, arg: float, /) -> None: ...

        @property
        def lightningRate(self) -> float: ...

        @lightningRate.setter
        def lightningRate(self, arg: float, /) -> None: ...

        @property
        def lightningLocalPointLightEnabled(self) -> bool: ...

        @lightningLocalPointLightEnabled.setter
        def lightningLocalPointLightEnabled(self, arg: bool, /) -> None: ...

        @property
        def lightningLocalPointDistance(self) -> float: ...

        @lightningLocalPointDistance.setter
        def lightningLocalPointDistance(self, arg: float, /) -> None: ...

        @property
        def lightningLocalPointRadius(self) -> float: ...

        @lightningLocalPointRadius.setter
        def lightningLocalPointRadius(self, arg: float, /) -> None: ...

        @property
        def lightningLocalPointIntensity(self) -> float: ...

        @lightningLocalPointIntensity.setter
        def lightningLocalPointIntensity(self, arg: float, /) -> None: ...

        @property
        def thunderSpeedOfSound(self) -> float: ...

        @thunderSpeedOfSound.setter
        def thunderSpeedOfSound(self, arg: float, /) -> None: ...

        @property
        def airTurbidity(self) -> float: ...

        @airTurbidity.setter
        def airTurbidity(self, arg: float, /) -> None: ...

        @property
        def groundAlbedo(self) -> float: ...

        @groundAlbedo.setter
        def groundAlbedo(self, arg: float, /) -> None: ...

        @property
        def useExplicitSunAngles(self) -> bool: ...

        @useExplicitSunAngles.setter
        def useExplicitSunAngles(self, arg: bool, /) -> None: ...

        @property
        def sunAzimuthDegrees(self) -> float: ...

        @sunAzimuthDegrees.setter
        def sunAzimuthDegrees(self, arg: float, /) -> None: ...

        @property
        def sunElevationDegrees(self) -> float: ...

        @sunElevationDegrees.setter
        def sunElevationDegrees(self, arg: float, /) -> None: ...

        @property
        def sunDiskSize(self) -> float: ...

        @sunDiskSize.setter
        def sunDiskSize(self, arg: float, /) -> None: ...

        @property
        def moonDiskSize(self) -> float: ...

        @moonDiskSize.setter
        def moonDiskSize(self, arg: float, /) -> None: ...

        @property
        def cloudAltitudeMeters(self) -> float: ...

        @cloudAltitudeMeters.setter
        def cloudAltitudeMeters(self, arg: float, /) -> None: ...

        @property
        def cloudThicknessMeters(self) -> float: ...

        @cloudThicknessMeters.setter
        def cloudThicknessMeters(self, arg: float, /) -> None: ...

        @property
        def cloudShadowStrength(self) -> float: ...

        @cloudShadowStrength.setter
        def cloudShadowStrength(self, arg: float, /) -> None: ...

        @property
        def cloudScale(self) -> float: ...

        @cloudScale.setter
        def cloudScale(self, arg: float, /) -> None: ...

        @property
        def cloudAnimationSpeed(self) -> float: ...

        @cloudAnimationSpeed.setter
        def cloudAnimationSpeed(self, arg: float, /) -> None: ...

        @property
        def lensDropletsEnabled(self) -> bool: ...

        @lensDropletsEnabled.setter
        def lensDropletsEnabled(self, arg: bool, /) -> None: ...

        @property
        def lensDropletStrength(self) -> float: ...

        @lensDropletStrength.setter
        def lensDropletStrength(self, arg: float, /) -> None: ...

    class WeatherDiagnostics:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def viewerEffectsActive(self) -> bool: ...

        @viewerEffectsActive.setter
        def viewerEffectsActive(self, arg: bool, /) -> None: ...

        @property
        def affectsSensors(self) -> bool: ...

        @affectsSensors.setter
        def affectsSensors(self, arg: bool, /) -> None: ...

        @property
        def preset(self) -> WeatherPreset: ...

        @preset.setter
        def preset(self, arg: WeatherPreset, /) -> None: ...

        @property
        def quality(self) -> WeatherQuality: ...

        @quality.setter
        def quality(self, arg: WeatherQuality, /) -> None: ...

        @property
        def seed(self) -> int: ...

        @seed.setter
        def seed(self, arg: int, /) -> None: ...

        @property
        def timeOfDayHours(self) -> float: ...

        @timeOfDayHours.setter
        def timeOfDayHours(self, arg: float, /) -> None: ...

        @property
        def latitude(self) -> float: ...

        @latitude.setter
        def latitude(self, arg: float, /) -> None: ...

        @property
        def longitude(self) -> float: ...

        @longitude.setter
        def longitude(self, arg: float, /) -> None: ...

        @property
        def automaticUtcOffset(self) -> bool: ...

        @automaticUtcOffset.setter
        def automaticUtcOffset(self, arg: bool, /) -> None: ...

        @property
        def utcOffsetHours(self) -> float: ...

        @utcOffsetHours.setter
        def utcOffsetHours(self, arg: float, /) -> None: ...

        @property
        def resolvedUtcOffsetHours(self) -> float: ...

        @resolvedUtcOffsetHours.setter
        def resolvedUtcOffsetHours(self, arg: float, /) -> None: ...

        @property
        def year(self) -> int: ...

        @year.setter
        def year(self, arg: int, /) -> None: ...

        @property
        def month(self) -> int: ...

        @month.setter
        def month(self, arg: int, /) -> None: ...

        @property
        def day(self) -> int: ...

        @day.setter
        def day(self, arg: int, /) -> None: ...

        @property
        def solarAzimuthDegrees(self) -> float: ...

        @solarAzimuthDegrees.setter
        def solarAzimuthDegrees(self, arg: float, /) -> None: ...

        @property
        def solarElevationDegrees(self) -> float: ...

        @solarElevationDegrees.setter
        def solarElevationDegrees(self, arg: float, /) -> None: ...

        @property
        def sunDirection(self) -> numpy.ndarray: ...

        @sunDirection.setter
        def sunDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def sunColor(self) -> numpy.ndarray: ...

        @sunColor.setter
        def sunColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def sunIntensity(self) -> float: ...

        @sunIntensity.setter
        def sunIntensity(self, arg: float, /) -> None: ...

        @property
        def moonLightActive(self) -> bool: ...

        @moonLightActive.setter
        def moonLightActive(self, arg: bool, /) -> None: ...

        @property
        def moonDirection(self) -> numpy.ndarray: ...

        @moonDirection.setter
        def moonDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def moonIntensity(self) -> float: ...

        @moonIntensity.setter
        def moonIntensity(self, arg: float, /) -> None: ...

        @property
        def skyExposure(self) -> float: ...

        @skyExposure.setter
        def skyExposure(self, arg: float, /) -> None: ...

        @property
        def cloudCoverage(self) -> float: ...

        @cloudCoverage.setter
        def cloudCoverage(self, arg: float, /) -> None: ...

        @property
        def cloudDensity(self) -> float: ...

        @cloudDensity.setter
        def cloudDensity(self, arg: float, /) -> None: ...

        @property
        def cloudLayerActive(self) -> bool: ...

        @cloudLayerActive.setter
        def cloudLayerActive(self, arg: bool, /) -> None: ...

        @property
        def cloudShadowActive(self) -> bool: ...

        @cloudShadowActive.setter
        def cloudShadowActive(self, arg: bool, /) -> None: ...

        @property
        def cloudAltitudeMeters(self) -> float: ...

        @cloudAltitudeMeters.setter
        def cloudAltitudeMeters(self, arg: float, /) -> None: ...

        @property
        def cloudThicknessMeters(self) -> float: ...

        @cloudThicknessMeters.setter
        def cloudThicknessMeters(self, arg: float, /) -> None: ...

        @property
        def cloudShadowStrength(self) -> float: ...

        @cloudShadowStrength.setter
        def cloudShadowStrength(self, arg: float, /) -> None: ...

        @property
        def cloudScale(self) -> float: ...

        @cloudScale.setter
        def cloudScale(self, arg: float, /) -> None: ...

        @property
        def cloudOffset(self) -> numpy.ndarray: ...

        @cloudOffset.setter
        def cloudOffset(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def fogDensity(self) -> float: ...

        @fogDensity.setter
        def fogDensity(self, arg: float, /) -> None: ...

        @property
        def visibilityMeters(self) -> float: ...

        @visibilityMeters.setter
        def visibilityMeters(self, arg: float, /) -> None: ...

        @property
        def heightFogActive(self) -> bool: ...

        @heightFogActive.setter
        def heightFogActive(self, arg: bool, /) -> None: ...

        @property
        def volumetricFogActive(self) -> bool: ...

        @volumetricFogActive.setter
        def volumetricFogActive(self, arg: bool, /) -> None: ...

        @property
        def heightFogDensity(self) -> float: ...

        @heightFogDensity.setter
        def heightFogDensity(self, arg: float, /) -> None: ...

        @property
        def heightFogBaseHeight(self) -> float: ...

        @heightFogBaseHeight.setter
        def heightFogBaseHeight(self, arg: float, /) -> None: ...

        @property
        def heightFogFalloff(self) -> float: ...

        @heightFogFalloff.setter
        def heightFogFalloff(self, arg: float, /) -> None: ...

        @property
        def fogColor(self) -> numpy.ndarray: ...

        @fogColor.setter
        def fogColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def fogAnisotropy(self) -> float: ...

        @fogAnisotropy.setter
        def fogAnisotropy(self, arg: float, /) -> None: ...

        @property
        def visibilityTransmittance100m(self) -> float: ...

        @visibilityTransmittance100m.setter
        def visibilityTransmittance100m(self, arg: float, /) -> None: ...

        @property
        def visibilityTransmittance1km(self) -> float: ...

        @visibilityTransmittance1km.setter
        def visibilityTransmittance1km(self, arg: float, /) -> None: ...

        @property
        def localFogVolumeBudget(self) -> int: ...

        @localFogVolumeBudget.setter
        def localFogVolumeBudget(self, arg: int, /) -> None: ...

        @property
        def localFogVolumeCount(self) -> int: ...

        @localFogVolumeCount.setter
        def localFogVolumeCount(self, arg: int, /) -> None: ...

        @property
        def localFogPositiveDensity(self) -> float: ...

        @localFogPositiveDensity.setter
        def localFogPositiveDensity(self, arg: float, /) -> None: ...

        @property
        def localFogNegativeDensity(self) -> float: ...

        @localFogNegativeDensity.setter
        def localFogNegativeDensity(self, arg: float, /) -> None: ...

        @property
        def precipitationRate(self) -> float: ...

        @precipitationRate.setter
        def precipitationRate(self, arg: float, /) -> None: ...

        @property
        def effectivePrecipitationRate(self) -> float: ...

        @effectivePrecipitationRate.setter
        def effectivePrecipitationRate(self, arg: float, /) -> None: ...

        @property
        def rainOcclusionStrength(self) -> float: ...

        @rainOcclusionStrength.setter
        def rainOcclusionStrength(self, arg: float, /) -> None: ...

        @property
        def wetness(self) -> float: ...

        @wetness.setter
        def wetness(self, arg: float, /) -> None: ...

        @property
        def wetnessAccumulationActive(self) -> bool: ...

        @wetnessAccumulationActive.setter
        def wetnessAccumulationActive(self, arg: bool, /) -> None: ...

        @property
        def accumulatedWetness(self) -> float: ...

        @accumulatedWetness.setter
        def accumulatedWetness(self, arg: float, /) -> None: ...

        @property
        def wetnessAccumulationRate(self) -> float: ...

        @wetnessAccumulationRate.setter
        def wetnessAccumulationRate(self, arg: float, /) -> None: ...

        @property
        def wetnessDryingRate(self) -> float: ...

        @wetnessDryingRate.setter
        def wetnessDryingRate(self, arg: float, /) -> None: ...

        @property
        def wetMaterialActive(self) -> bool: ...

        @wetMaterialActive.setter
        def wetMaterialActive(self, arg: bool, /) -> None: ...

        @property
        def wetMaterialStrength(self) -> float: ...

        @wetMaterialStrength.setter
        def wetMaterialStrength(self, arg: float, /) -> None: ...

        @property
        def puddleStrength(self) -> float: ...

        @puddleStrength.setter
        def puddleStrength(self, arg: float, /) -> None: ...

        @property
        def rainRippleStrength(self) -> float: ...

        @rainRippleStrength.setter
        def rainRippleStrength(self, arg: float, /) -> None: ...

        @property
        def wetAlbedoDarkening(self) -> float: ...

        @wetAlbedoDarkening.setter
        def wetAlbedoDarkening(self, arg: float, /) -> None: ...

        @property
        def wetRoughnessScale(self) -> float: ...

        @wetRoughnessScale.setter
        def wetRoughnessScale(self, arg: float, /) -> None: ...

        @property
        def wetSpecularBoost(self) -> float: ...

        @wetSpecularBoost.setter
        def wetSpecularBoost(self, arg: float, /) -> None: ...

        @property
        def snowCoverage(self) -> float: ...

        @snowCoverage.setter
        def snowCoverage(self, arg: float, /) -> None: ...

        @property
        def snowMeltActive(self) -> bool: ...

        @snowMeltActive.setter
        def snowMeltActive(self, arg: bool, /) -> None: ...

        @property
        def snowMeltFactor(self) -> float: ...

        @snowMeltFactor.setter
        def snowMeltFactor(self, arg: float, /) -> None: ...

        @property
        def snowMeltWetnessBoost(self) -> float: ...

        @snowMeltWetnessBoost.setter
        def snowMeltWetnessBoost(self, arg: float, /) -> None: ...

        @property
        def snowMaterialActive(self) -> bool: ...

        @snowMaterialActive.setter
        def snowMaterialActive(self, arg: bool, /) -> None: ...

        @property
        def snowMaterialStrength(self) -> float: ...

        @snowMaterialStrength.setter
        def snowMaterialStrength(self, arg: float, /) -> None: ...

        @property
        def snowAccumulationStrength(self) -> float: ...

        @snowAccumulationStrength.setter
        def snowAccumulationStrength(self, arg: float, /) -> None: ...

        @property
        def snowAlbedoBlend(self) -> float: ...

        @snowAlbedoBlend.setter
        def snowAlbedoBlend(self, arg: float, /) -> None: ...

        @property
        def snowRoughness(self) -> float: ...

        @snowRoughness.setter
        def snowRoughness(self, arg: float, /) -> None: ...

        @property
        def snowMetallicScale(self) -> float: ...

        @snowMetallicScale.setter
        def snowMetallicScale(self, arg: float, /) -> None: ...

        @property
        def snowNormalSoftening(self) -> float: ...

        @snowNormalSoftening.setter
        def snowNormalSoftening(self, arg: float, /) -> None: ...

        @property
        def lightningRate(self) -> float: ...

        @lightningRate.setter
        def lightningRate(self, arg: float, /) -> None: ...

        @property
        def lightningActive(self) -> bool: ...

        @lightningActive.setter
        def lightningActive(self, arg: bool, /) -> None: ...

        @property
        def lightningEventIndex(self) -> int: ...

        @lightningEventIndex.setter
        def lightningEventIndex(self, arg: int, /) -> None: ...

        @property
        def lightningEventTimeSeconds(self) -> float: ...

        @lightningEventTimeSeconds.setter
        def lightningEventTimeSeconds(self, arg: float, /) -> None: ...

        @property
        def lightningNextEventSeconds(self) -> float: ...

        @lightningNextEventSeconds.setter
        def lightningNextEventSeconds(self, arg: float, /) -> None: ...

        @property
        def lightningEventPhaseSeconds(self) -> float: ...

        @lightningEventPhaseSeconds.setter
        def lightningEventPhaseSeconds(self, arg: float, /) -> None: ...

        @property
        def lightningEventDurationSeconds(self) -> float: ...

        @lightningEventDurationSeconds.setter
        def lightningEventDurationSeconds(self, arg: float, /) -> None: ...

        @property
        def lightningIntensity(self) -> float: ...

        @lightningIntensity.setter
        def lightningIntensity(self, arg: float, /) -> None: ...

        @property
        def lightningExposureBoost(self) -> float: ...

        @lightningExposureBoost.setter
        def lightningExposureBoost(self, arg: float, /) -> None: ...

        @property
        def lightningSkyFlash(self) -> float: ...

        @lightningSkyFlash.setter
        def lightningSkyFlash(self, arg: float, /) -> None: ...

        @property
        def lightningDirection(self) -> numpy.ndarray: ...

        @lightningDirection.setter
        def lightningDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lightningColor(self) -> numpy.ndarray: ...

        @lightningColor.setter
        def lightningColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lightningPointLightActive(self) -> bool: ...

        @lightningPointLightActive.setter
        def lightningPointLightActive(self, arg: bool, /) -> None: ...

        @property
        def lightningPointLightPosition(self) -> numpy.ndarray: ...

        @lightningPointLightPosition.setter
        def lightningPointLightPosition(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lightningPointLightColor(self) -> numpy.ndarray: ...

        @lightningPointLightColor.setter
        def lightningPointLightColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lightningPointLightRadius(self) -> float: ...

        @lightningPointLightRadius.setter
        def lightningPointLightRadius(self, arg: float, /) -> None: ...

        @property
        def lightningPointLightIntensity(self) -> float: ...

        @lightningPointLightIntensity.setter
        def lightningPointLightIntensity(self, arg: float, /) -> None: ...

        @property
        def thunderDelaySeconds(self) -> float: ...

        @thunderDelaySeconds.setter
        def thunderDelaySeconds(self, arg: float, /) -> None: ...

        @property
        def precipitationParticlesActive(self) -> bool: ...

        @precipitationParticlesActive.setter
        def precipitationParticlesActive(self, arg: bool, /) -> None: ...

        @property
        def snowParticlesActive(self) -> bool: ...

        @snowParticlesActive.setter
        def snowParticlesActive(self, arg: bool, /) -> None: ...

        @property
        def rainParticlesActive(self) -> bool: ...

        @rainParticlesActive.setter
        def rainParticlesActive(self, arg: bool, /) -> None: ...

        @property
        def precipitationParticleBudget(self) -> int: ...

        @precipitationParticleBudget.setter
        def precipitationParticleBudget(self, arg: int, /) -> None: ...

        @property
        def precipitationParticlesRendered(self) -> int: ...

        @precipitationParticlesRendered.setter
        def precipitationParticlesRendered(self, arg: int, /) -> None: ...

        @property
        def precipitationParticleSizePixels(self) -> float: ...

        @precipitationParticleSizePixels.setter
        def precipitationParticleSizePixels(self, arg: float, /) -> None: ...

        @property
        def precipitationParticleDepthMeters(self) -> float: ...

        @precipitationParticleDepthMeters.setter
        def precipitationParticleDepthMeters(self, arg: float, /) -> None: ...

        @property
        def precipitationParticleAlpha(self) -> float: ...

        @precipitationParticleAlpha.setter
        def precipitationParticleAlpha(self, arg: float, /) -> None: ...

        @property
        def precipitationParticleColor(self) -> numpy.ndarray: ...

        @precipitationParticleColor.setter
        def precipitationParticleColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def rainSplashParticlesActive(self) -> bool: ...

        @rainSplashParticlesActive.setter
        def rainSplashParticlesActive(self, arg: bool, /) -> None: ...

        @property
        def rainSplashParticlesRendered(self) -> int: ...

        @rainSplashParticlesRendered.setter
        def rainSplashParticlesRendered(self, arg: int, /) -> None: ...

        @property
        def rainSplashParticleSizePixels(self) -> float: ...

        @rainSplashParticleSizePixels.setter
        def rainSplashParticleSizePixels(self, arg: float, /) -> None: ...

        @property
        def rainSplashParticleAlpha(self) -> float: ...

        @rainSplashParticleAlpha.setter
        def rainSplashParticleAlpha(self, arg: float, /) -> None: ...

        @property
        def rainSplashParticleColor(self) -> numpy.ndarray: ...

        @rainSplashParticleColor.setter
        def rainSplashParticleColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def lensDropletsActive(self) -> bool: ...

        @lensDropletsActive.setter
        def lensDropletsActive(self, arg: bool, /) -> None: ...

        @property
        def lensDropletsRendered(self) -> int: ...

        @lensDropletsRendered.setter
        def lensDropletsRendered(self, arg: int, /) -> None: ...

        @property
        def lensDropletStrength(self) -> float: ...

        @lensDropletStrength.setter
        def lensDropletStrength(self, arg: float, /) -> None: ...

        @property
        def lensDropletMaxSizePixels(self) -> float: ...

        @lensDropletMaxSizePixels.setter
        def lensDropletMaxSizePixels(self, arg: float, /) -> None: ...

        @property
        def lensDropletAlpha(self) -> float: ...

        @lensDropletAlpha.setter
        def lensDropletAlpha(self, arg: float, /) -> None: ...

        @property
        def weatherTextureBytes(self) -> int: ...

        @weatherTextureBytes.setter
        def weatherTextureBytes(self, arg: int, /) -> None: ...

        @property
        def transitionActive(self) -> bool: ...

        @transitionActive.setter
        def transitionActive(self, arg: bool, /) -> None: ...

        @property
        def transitionAlpha(self) -> float: ...

        @transitionAlpha.setter
        def transitionAlpha(self, arg: float, /) -> None: ...

        @property
        def skyEnvironmentValid(self) -> bool: ...

        @skyEnvironmentValid.setter
        def skyEnvironmentValid(self, arg: bool, /) -> None: ...

        @property
        def skyEnvironmentMap(self) -> int: ...

        @skyEnvironmentMap.setter
        def skyEnvironmentMap(self, arg: int, /) -> None: ...

        @property
        def skyIrradianceMap(self) -> int: ...

        @skyIrradianceMap.setter
        def skyIrradianceMap(self, arg: int, /) -> None: ...

        @property
        def skyEnvironmentFaceSize(self) -> int: ...

        @skyEnvironmentFaceSize.setter
        def skyEnvironmentFaceSize(self, arg: int, /) -> None: ...

        @property
        def skyIrradianceFaceSize(self) -> int: ...

        @skyIrradianceFaceSize.setter
        def skyIrradianceFaceSize(self, arg: int, /) -> None: ...

        @property
        def skyEnvironmentGenerationCount(self) -> int: ...

        @skyEnvironmentGenerationCount.setter
        def skyEnvironmentGenerationCount(self, arg: int, /) -> None: ...

    class WeatherSkyEnvironment:
        def __init__(self) -> None: ...

        @property
        def environmentMap(self) -> int: ...

        @environmentMap.setter
        def environmentMap(self, arg: int, /) -> None: ...

        @property
        def irradianceMap(self) -> int: ...

        @irradianceMap.setter
        def irradianceMap(self, arg: int, /) -> None: ...

        @property
        def environmentFaceSize(self) -> int: ...

        @environmentFaceSize.setter
        def environmentFaceSize(self, arg: int, /) -> None: ...

        @property
        def irradianceFaceSize(self) -> int: ...

        @irradianceFaceSize.setter
        def irradianceFaceSize(self, arg: int, /) -> None: ...

        @property
        def generationCount(self) -> int: ...

        @generationCount.setter
        def generationCount(self, arg: int, /) -> None: ...

        @property
        def valid(self) -> bool: ...

        @valid.setter
        def valid(self, arg: bool, /) -> None: ...

    class ProjectedDecal:
        def __init__(self) -> None: ...

        @property
        def center(self) -> numpy.ndarray: ...

        @center.setter
        def center(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def halfExtents(self) -> numpy.ndarray: ...

        @halfExtents.setter
        def halfExtents(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def orientation(self) -> numpy.ndarray: ...

        @orientation.setter
        def orientation(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def color(self) -> numpy.ndarray: ...

        @color.setter
        def color(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def edgeFade(self) -> float: ...

        @edgeFade.setter
        def edgeFade(self, arg: float, /) -> None: ...

        @property
        def normalFade(self) -> float: ...

        @normalFade.setter
        def normalFade(self, arg: float, /) -> None: ...

        @property
        def upperFade(self) -> float: ...

        @upperFade.setter
        def upperFade(self, arg: float, /) -> None: ...

        @property
        def lowerFade(self) -> float: ...

        @lowerFade.setter
        def lowerFade(self, arg: float, /) -> None: ...

        @property
        def albedoMap(self) -> int: ...

        @albedoMap.setter
        def albedoMap(self, arg: int, /) -> None: ...

        @property
        def emissionMap(self) -> int: ...

        @emissionMap.setter
        def emissionMap(self, arg: int, /) -> None: ...

        @property
        def normalMap(self) -> int: ...

        @normalMap.setter
        def normalMap(self, arg: int, /) -> None: ...

        @property
        def ormMap(self) -> int: ...

        @ormMap.setter
        def ormMap(self, arg: int, /) -> None: ...

        @property
        def textureUvScaleOffset(self) -> numpy.ndarray: ...

        @textureUvScaleOffset.setter
        def textureUvScaleOffset(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def albedoMix(self) -> float: ...

        @albedoMix.setter
        def albedoMix(self, arg: float, /) -> None: ...

        @property
        def emissionEnergy(self) -> float: ...

        @emissionEnergy.setter
        def emissionEnergy(self, arg: float, /) -> None: ...

        @property
        def normalStrength(self) -> float: ...

        @normalStrength.setter
        def normalStrength(self, arg: float, /) -> None: ...

        @property
        def ormStrength(self) -> float: ...

        @ormStrength.setter
        def ormStrength(self, arg: float, /) -> None: ...

        @property
        def distanceFadeEnabled(self) -> bool: ...

        @distanceFadeEnabled.setter
        def distanceFadeEnabled(self, arg: bool, /) -> None: ...

        @property
        def distanceFadeBegin(self) -> float: ...

        @distanceFadeBegin.setter
        def distanceFadeBegin(self, arg: float, /) -> None: ...

        @property
        def distanceFadeLength(self) -> float: ...

        @distanceFadeLength.setter
        def distanceFadeLength(self, arg: float, /) -> None: ...

    class IrradianceVolume:
        def __init__(self) -> None: ...

        @property
        def center(self) -> numpy.ndarray: ...

        @center.setter
        def center(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def halfExtents(self) -> numpy.ndarray: ...

        @halfExtents.setter
        def halfExtents(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def orientation(self) -> numpy.ndarray: ...

        @orientation.setter
        def orientation(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def color(self) -> numpy.ndarray: ...

        @color.setter
        def color(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def strength(self) -> float: ...

        @strength.setter
        def strength(self, arg: float, /) -> None: ...

        @property
        def edgeFade(self) -> float: ...

        @edgeFade.setter
        def edgeFade(self, arg: float, /) -> None: ...

        @property
        def normalBias(self) -> float: ...

        @normalBias.setter
        def normalBias(self, arg: float, /) -> None: ...

    kMaxAdditionalLights: int = 16

    kMaxAdditionalShadowLights: int = 8

    kMaxLocalFogVolumes: int = 8

    kMaxProjectedDecals: int = 8

    kMaxIrradianceVolumes: int = 8

    def threadingMode(self) -> RayraiWindow.ThreadingMode:
        """
        /**
           * @brief Threading mode this renderer was constructed with.
           * @return The mode passed to the constructor.
           */
        C++: ThreadingMode () const
        """

    @staticmethod
    def loadTextureWithTiling(path: str, useSrgbFormat: bool = False) -> int:
        """
        /**
           * @brief Load an image file into an OpenGL texture with repeat tiling.
           * @param path Path to an image file supported by stb_image.
           * @param useSrgbFormat If true, RGB/RGBA textures are uploaded as sRGB color textures.
           * Use true for albedo/emissive color maps and false for data maps such as normal,
           * metallic-roughness, AO, depth, and masks.
           * @return OpenGL texture id, or 0 on failure.
           *
           * The texture gets GL_REPEAT wrapping, mipmaps, and the process-wide defaults
           * from setDefaultTextureFilteringQuality(). Results are cached per thread and
           * per GL context by (path, sRGB flag), so repeated calls return the same id;
           * a failed load is cached as 0 too. Requires a current GL context.
           */
        C++: unsigned int (const char *, bool)
        """

    def loadSceneHdrCubemapCached(self, path: str, faceSize: int) -> int:
        """
        /// Load a scene HDR map once per path, file signature and face size. The viewer owns
          /// the texture until destruction; callers must not delete it. GL context required.
        C++: unsigned int (const std::string &, int)
        """

    def acquireSceneHdrCubemapCached(self, path: str, faceSize: int) -> TextureHandle:
        """
        /// Acquire a scene HDR texture while it is installed in a background or visual.
          /// Unused scene entries are evicted; the viewer frees textures before context teardown.
        C++: std::shared_ptr<const unsigned int> (const std::string &, int)
        """

    @staticmethod
    def loadHdrEquirectangularCubemap(path: str, faceSize: int = 128, generateMipmaps: bool = True) -> int:
        """
        /**
           * @brief Load a Radiance .hdr equirectangular environment into a cubemap texture.
           *
           * This is intended for High/Ultra viewer PBR workflows. It does not allocate anything unless
           * the application explicitly calls it.
           *
           * @param path Path to a .hdr image supported by stb_image.
           * @param faceSize Cubemap face size in pixels.
           * @param generateMipmaps Whether to generate mip levels for roughness sampling.
           * @return OpenGL cubemap id, or 0 on failure.
           */
        C++: unsigned int (const char *, int, bool)
        """

    @staticmethod
    def createHdrIrradianceCubemap(path: str, faceSize: int = 32, sampleCount: int = 256) -> int:
        """
        /**
           * @brief Generate a diffuse irradiance cubemap from a Radiance .hdr equirectangular environment.
           *
           * This is CPU-side cosine-hemisphere integration intended for High/Ultra PBR setup. Creation is
           * explicit so Fast/Balanced RL rendering does not pay the cost.
           *
           * @param path Path to a .hdr image supported by stb_image.
           * @param faceSize Irradiance cubemap face size in pixels.
           * @param sampleCount Number of hemisphere samples per texel.
           * @return OpenGL cubemap id, or 0 on failure.
           */
        C++: unsigned int (const char *, int, int)
        """

    @staticmethod
    def createHdrPrefilteredEnvironmentCubemap(path: str, faceSize: int = 128, mipLevels: int = 5, sampleCount: int = 256) -> int:
        """
        /**
           * @brief Generate a prefiltered specular environment cubemap from a Radiance .hdr equirectangular environment.
           *
           * Mip level 0 is sharp and higher mip levels are GGX-prefiltered with increasing roughness. Use
           * this as Material::prefilteredEnvironmentMap together with createSplitSumBrdfLut().
           *
           * @param path Path to a .hdr image supported by stb_image.
           * @param faceSize Base cubemap face size in pixels.
           * @param mipLevels Number of mip levels to allocate and integrate.
           * @param sampleCount Number of GGX samples per texel.
           * @return OpenGL cubemap id, or 0 on failure.
           */
        C++: unsigned int (const char *, int, int, int)
        """

    @staticmethod
    def createSplitSumBrdfLut(size: int = 256, sampleCount: int = 512) -> int:
        """
        /**
           * @brief Create a numerically integrated split-sum BRDF LUT.
           *
           * Use this as Material::brdfLut for High/Ultra PBR. Creation is explicit so Fast/Balance paths
           * do not pay for it. The result is an RG16F texture (x = N.V, y = roughness).
           *
           * @param size LUT width and height in texels.
           * @param sampleCount GGX samples per texel.
           * @return OpenGL texture id owned by the caller, or 0 if either argument is not positive.
           */
        C++: unsigned int (int, int)
        """

    @staticmethod
    def setTextureAnisotropy(target: int, textureId: int, requestedLevel: float) -> None:
        """
        /**
           * @brief Apply anisotropic filtering to an existing 2D or cubemap texture if supported.
           * @param target Texture target such as GL_TEXTURE_2D or GL_TEXTURE_CUBE_MAP.
           * @param textureId OpenGL texture id.
           * @param requestedLevel Requested anisotropy level. Values <= 1 leave the texture unchanged.
           */
        C++: void (unsigned int, unsigned int, float)
        """

    @staticmethod
    def setTextureFilteringQuality(target: int, textureId: int, anisotropy: float, mipLodBias: float) -> None:
        """
        /**
           * @brief Apply minification quality controls to an existing texture.
           *
           * Positive mip LOD bias prefers blurrier mips and reduces distant texture shimmer. Negative
           * values sharpen but can alias, so RayRai clamps the bias to [-1, 2].
           *
           * @param target Texture target such as GL_TEXTURE_2D.
           * @param textureId OpenGL texture id; 0 is ignored.
           * @param anisotropy Requested anisotropy, see setTextureAnisotropy().
           * @param mipLodBias Mip LOD bias, clamped to [-1, 2].
           */
        C++: void (unsigned int, unsigned int, float, float)
        """

    @staticmethod
    def setDefaultTextureFilteringQuality(anisotropy: float, mipLodBias: float) -> None:
        """
        /**
           * @brief Set process-wide filtering defaults used by future loadTextureWithTiling() calls.
           * @param anisotropy Default anisotropy, clamped to [1, 16].
           * @param mipLodBias Default mip LOD bias, clamped to [-1, 2].
           */
        C++: void (float, float)
        """

    @staticmethod
    def getDefaultTextureFilteringQuality() -> tuple:
        """
        /**
           * @brief Read process-wide filtering defaults used by future loadTextureWithTiling() calls.
           * @param[out] anisotropy Default anisotropy.
           * @param[out] mipLodBias Default mip LOD bias.
           */
        C++: void (float &, float &)
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def applyTextureFilteringQualityToLoadedTextures(self) -> int:
        """
        /**
           * @brief Apply current texture filtering quality to currently loaded RayRai visual textures.
           *
           * Uses RenderQualitySettings::textureAnisotropy and textureMipLodBias and covers the
           * ground texture, custom visuals and instanced visuals.
           * @return Number of texture ids updated.
           */
        C++: size_t ()
        """

    @staticmethod
    def setTextureSamplerState(target: int, textureId: int, wrapS: int, wrapT: int, minFilter: int, magFilter: int) -> None:
        """
        /**
           * @brief Apply sampler wrap/filter state to an existing texture.
           * @param target Texture target such as GL_TEXTURE_2D.
           * @param textureId OpenGL texture id; 0 is ignored.
           * @param wrapS GL wrap mode for S (e.g. GL_REPEAT).
           * @param wrapT GL wrap mode for T.
           * @param minFilter GL minification filter.
           * @param magFilter GL magnification filter.
           */
        C++: void (unsigned int, unsigned int, int, int, int, int)
        """

    @staticmethod
    def loadColorTextureWithTiling(path: str) -> int:
        """
        /**
           * @brief Load an albedo/emissive color texture as sRGB.
           * @param path Path to an image file supported by stb_image.
           * @return OpenGL texture id, or 0 on failure.
           */
        C++: unsigned int (const char *)
        """

    @staticmethod
    def loadDataTextureWithTiling(path: str) -> int:
        """
        /**
           * @brief Load a data texture as linear values.
           * @param path Path to an image file supported by stb_image.
           * @return OpenGL texture id, or 0 on failure.
           */
        C++: unsigned int (const char *)
        """

    def setAsyncMeshLoadingEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Enable or disable async mesh file import for future mesh visuals/world assets.
           *
           * Async loading parses files and prepares base geometry and instanced LODs on workers.
           * Pending instances are skipped; textures and incremental GPU uploads use the render thread. Disable it when a caller needs mesh
           * assets to be available immediately after addVisualMesh()/importVisualScene().
           * Disabling it also finishes the meshes still loading before it returns, for example
           * those of the world's scene that the constructor applied; call it with this window's
           * GL context current. To load only the meshes requested next synchronously, use a
           * SynchronousMeshLoadingScope instead.
           * Enabled by default; USD files are always imported synchronously.
           * @param enabled True to import future mesh files asynchronously; false to load them
           *        synchronously and finish the pending ones.
           */
        C++: void (bool)
        """

    def asyncMeshLoadingEnabled(self) -> bool:
        """
        /**
           * @brief True when future mesh file imports are started asynchronously.
           * @return Current async-import flag.
           */
        C++: bool () const
        """

    def pollAsyncMeshLoads(self, maxAssets: int = 1) -> int:
        """
        /**
           * @brief Finalize ready async mesh imports on the render thread.
           * @param maxAssets Maximum import/upload stages to advance (zero does no work).
           * Instanced LOD preparation is also polled, with one submesh upload per batch.
           * Requires the window's GL context: with another current, the pending uploads
           * give their staging space back and this throws std::logic_error before any
           * advances (as does a frame's poll in update()).
           * @return Number of assets finalized.
           */
        C++: size_t (size_t)
        """

    def pendingAsyncMeshLoadCount(self) -> int:
        """
        /**
           * @brief Pending imports plus instanced LOD jobs, including unfinished GPU uploads, and the
           * HDR background of the world's scene while it loads.
           * @return Number of pending jobs; 0 once everything has been finalized by pollAsyncMeshLoads().
           */
        C++: size_t () const
        """

    def debugAssets(self) -> RayraiGlobalAsset:
        """
        /**
           * @brief The asset manager this renderer imports through.
           *
           * Exposed so tests can inspect the meshes this renderer actually holds
           * rather than re-deriving them through a second manager, which only agrees
           * when the asset happens to be eligible for the process-wide GPU cache.
           * @return Reference owned by this renderer, valid for its lifetime.
           */
        C++: RayraiGlobalAsset &() const
        """

    def setKeyboardShortcutsEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Whether this renderer reacts to global keyboard shortcuts.
           *
           * The renderer's built-in shortcuts read SDL's keyboard state, which is one
           * array for the whole process and carries no notion of which window has
           * focus. A process running several renderers would otherwise see a key press
           * act on all of them at once, so a multi-window application should leave this
           * on only for the focused renderer. Defaults to true, which is the behaviour
           * of a single-renderer application. Currently the only shortcut is `B`, which
           * toggles the picking-buffer debug view. Camera movement keys are controlled
           * separately by setKeyboardInputEnabled().
           * @param enabled False to ignore shortcut keys in this renderer.
           */
        C++: void (bool)
        """

    def keyboardShortcutsEnabled(self) -> bool:
        """
        /// @return True when this renderer reacts to global keyboard shortcuts (default true).
        C++: bool () const
        """

    def pbrDebugOutputMode(self) -> int:
        """
        /**
           * @brief PBR debug output mode in force for this renderer.
           *
           * The environment variable RAYRAI_PBR_DEBUG_OUTPUT sets it process-wide;
           * captureDebugPasses overrides it for the frames it renders. The override
           * belongs to this renderer: rewriting the process environment raced every
           * other worker's getenv on the render path, and a thread-local override
           * would follow whichever thread happened to drive the renderer.
           * @return Active mode; 0 is normal shading.
           */
        C++: int () const
        """

    def setPbrDebugOutputMode(self, mode: int) -> None:
        """
        /// Select a renderer-local material capture mode (0..31). -1 restores the
          /// process default. Changing the mode invalidates cached scene color.
          /// @param mode Mode in [-1, 31]; throws std::invalid_argument otherwise.
        C++: void (int)
        """

    @staticmethod
    def prewarmShadersForCurrentContext(threadingMode: RayraiWindow.ThreadingMode = RayraiWindow.ThreadingMode.MultiThread, shaderBinaryCacheDirectory: str = '') -> None:
        """
        /**
           * @brief Compile every built-in shader using the GL context currently bound
           *        to the calling thread, populating the on-disk binary cache.
           *
           * Intended for parallel RL workflows: spawn a background thread that holds
           * its own offscreen GL context, call this once, and worker threads that
           * later construct `RayraiWindow` instances will find the heavy shaders
           * already in the cache (or briefly block on the per-shader compile mutex
           * for any program this thread is still finishing).
           *
           * Net effect across a process: each shader's GLSL→binary compile runs
           * exactly once, regardless of how many worker threads concurrently
           * construct `RayraiWindow`s.
           *
           * Pre-conditions: the caller's thread has made a valid offscreen GL
           * context current. Post-condition: the temporary `RayraiWindow` used for
           * compilation is destroyed before this returns; the caller's context is
           * left current. The caller is responsible for the surrounding
           * `SDL_GL_MakeCurrent(nullptr)` / `SDL_GL_DeleteContext` / thread join.
           *
           * Side effect: the temporary renderer re-applies the process-wide shader
           * settings (one compile thread, binary cache enabled at
           * @p shaderBinaryCacheDirectory, cache logging off), like any constructor.
           *
           * @param threadingMode Must match the mode used by the worker threads that
           *        will eventually construct their own `RayraiWindow`s. Defaults to
           *        `MultiThread` since prewarming only makes sense when other
           *        threads will also render.
           * @param shaderBinaryCacheDirectory Optional explicit cache directory.
           *        Empty selects the default per-user cache directory (see the
           *        RayraiWindow constructor).
           */
        C++: void (ThreadingMode, std::string)
        """

    @overload
    def resize(self, newWidth: int, newHeight: int) -> None:
        """
        /**
           * @brief Resize the internal camera render target.
           *
           * Non-positive or unchanged sizes are ignored. update() calls this with its
           * own size every frame, so explicit calls are only needed before rendering
           * through other paths.
           * @param newWidth New framebuffer width in pixels.
           * @param newHeight New framebuffer height in pixels.
           */
        C++: void (int, int)
        """

    @overload
    def resize(self, width: int, height: int) -> None: ...

    def update(self, w: int, h: int, isHovering: bool, cX: int, cY: int, shouldClick: bool) -> None:
        """
        /**
           * @brief Render one frame into the internal camera's offscreen texture.
           *
           * The interactive viewer path: resizes the target to @p w x @p h, finalizes at
           * most one pending async mesh import, syncs raisim objects, applies mouse and
           * keyboard camera input, then renders shadows, the scene and post-processing.
           * Read the result with getImageTexture(). Requires this renderer's GL context.
           *
           * @param w Target width in pixels.
           * @param h Target height in pixels.
           * @param isHovering Whether the mouse is over the render area; mouse input is ignored otherwise.
           * @param cX Cursor x position inside the render area, in pixels.
           * @param cY Cursor y position inside the render area, in pixels.
           * @param shouldClick Whether mouse click selection should be processed.
           */
        C++: void (int, int, bool, int, int, bool)
        """

    def cancelViewportMouseDrag(self) -> None:
        """
        /** @brief Stop an in-progress viewport mouse drag without changing button state. */
        C++: void ()
        """

    def setKeyboardInputEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Allow or block this renderer's WASD/Space camera movement.
           * @param enabled False to ignore the keyboard for this renderer.
           *
           * The camera reads the process-wide SDL keyboard state, which is shared by
           * every renderer in the process. A host that draws more than one view at a
           * time (the TCP viewer's split panes) must therefore leave this on for just
           * one of them, or a single key press walks every camera at once. Mouse input
           * needs no such switch: it is already scoped by the hover flag passed to
           * update().
           */
        C++: void (bool)
        """

    def keyboardInputEnabled(self) -> bool:
        """
        /** @return True when this renderer acts on camera keys (default true). */
        C++: bool () const
        """

    def getImageTexture(self) -> int:
        """
        /**
           * @brief Return the GL texture id of the internal camera output.
           * @return Texture id for the most recent rendered frame, owned by the renderer
           *         (it can change after resize()).
           */
        C++: unsigned int () const
        """

    def setCameraFixedDistance(self, fixed: bool) -> None:
        """
        /**
           * @brief Keep the camera distance constant when orbiting a target.
           * @param fixed True to lock distance.
           */
        C++: void (bool)
        """

    def isCameraFixedDistance(self) -> bool:
        """
        /** @brief True when the orbit distance is locked.
           *  @return Lock state. */
        C++: bool () const
        """

    def cameraFixedDistance(self) -> float:
        """
        /** @brief Current locked orbit distance (metres).
           *  @return Distance used while the orbit distance is locked. */
        C++: float () const
        """

    def setCameraFixedTarget(self, fixed: bool) -> None:
        """
        /**
           * @brief Keep the camera target fixed in world space.
           * @param fixed True to lock the target position.
           */
        C++: void (bool)
        """

    def isCameraFixedTarget(self) -> bool:
        """
        /** @brief True when the camera target is locked to a fixed point.
           *  @return Lock state. */
        C++: bool () const
        """

    def setBackgroundColor(self, color: numpy.ndarray) -> None:
        """
        /**
           * @brief Set background clear color using legacy 0..255 RGBA channels.
           * @param color RGBA color where each channel is in [0, 255].
           *
           * Identical to setBackgroundColorRgb255(); prefer that name or
           * setBackgroundColorLinear() in new code. Overwritten by
           * setRenderQualitySettings()/setRenderQualityPreset().
           */
        C++: void (const glm::vec4 &)
        """

    def setBackgroundColorRgb255(self, color: numpy.ndarray) -> None:
        """
        /**
           * @brief Set background clear color using 0..255 RGBA channels.
           * @param color RGBA color where each channel is in [0, 255].
           */
        C++: void (const glm::vec4 &)
        """

    def setBackgroundColorLinear(self, color: numpy.ndarray) -> None:
        """
        /**
           * @brief Set background clear color using linear 0..1 RGBA channels.
           * @param color Linear RGBA color where each channel is in [0, 1].
           */
        C++: void (const glm::vec4 &)
        """

    def getBackgroundColor(self) -> numpy.ndarray:
        """
        /**
           * @brief Get the current background clear color in 0..255 RGBA channels.
           *
           * Mirrors `setBackgroundColor*()` so the value round-trips through
           * serialization without a separate Camera lookup.
           * @return RGBA color, each channel in [0, 255].
           */
        C++: glm::vec4 () const
        """

    def getBackgroundColorLinear(self) -> numpy.ndarray:
        """
        /**
           * @brief Get the current background clear color in linear 0..1 RGBA channels.
           * @return Linear RGBA color.
           */
        C++: glm::vec4 () const
        """

    def setEnvironmentBackground(self, environmentMap: int, exposure: float = 1.0) -> None:
        """
        /**
           * @brief Render a cubemap behind the interactive viewer camera.
           * @param environmentMap Cubemap texture id, or 0 to disable the environment background.
           *        Not owned: the caller keeps it alive while it is installed.
           * @param exposure Multiplier applied while drawing the environment background.
           */
        C++: void (unsigned int, float)
        """

    def setSceneEnvironmentBackground(self, texture: TextureHandle, exposure: float = 1.0) -> None:
        """
        /// Install a background while retaining its scene HDR texture lease.
        C++: void (std::shared_ptr<const unsigned int>, float)
        """

    def environmentBackgroundMap(self) -> int:
        """
        /**
           * @brief Current interactive viewer environment background cubemap.
           * @return Cubemap id, or 0 if disabled.
           */
        C++: unsigned int () const
        """

    def environmentBackgroundExposure(self) -> float:
        """
        /**
           * @brief Current interactive viewer environment background exposure.
           * @return Exposure multiplier.
           */
        C++: float () const
        """

    def clearEnvironmentBackground(self) -> None:
        """
        /**
           * @brief Disable the interactive viewer environment background.
           */
        C++: void ()
        """

    def setFogDensity(self, density: float) -> None:
        """
        /**
           * @brief Set scene fog density used by the mesh shader.
           *
           * Overwritten by setRenderQualitySettings() (RenderQualitySettings::fogDensity).
           * @param density Fog density in 1/m (default 0.01); negative values are clamped to 0.
           */
        C++: void (float)
        """

    def getFogDensity(self) -> float:
        """
        /**
           * @brief Get current fog density.
           * @return Fog density value.
           */
        C++: float () const
        """

    def getCamera(self) -> Camera:
        """
        /**
           * @brief Access the internal camera for direct manipulation.
           * Targets are allocated on first render or explicit ensureRenderTargets().
           * @return Reference to the internal camera.
           */
        C++: Camera &()
        """

    @overload
    def getTargetVisual(self) -> Visuals | None:
        """
        /**
           * @brief Get the currently selected target visual, if any.
           * @return Pointer to the target visual or nullptr.
           */
        C++: const Visuals *() const
        """

    @overload
    def getTargetVisual(self) -> Visuals | None:
        """
        /// @copydoc getTargetVisual() const
        C++: Visuals *()
        """

    def pickTargetVisualAt(self, x: int, y: int) -> Visuals | None:
        """
        /**
           * @brief Pick a custom visual at a pixel of the internal camera.
           *
           * Runs a picking pass immediately (GL context required). Only visuals added
           * through this renderer can be returned.
           * @param x Pixel x coordinate in the internal camera's framebuffer (top-left origin).
           * @param y Pixel y coordinate, growing downward; out-of-range pixels pick nothing.
           * @return Pointer to the picked visual (owned by the renderer) or nullptr.
           */
        C++: Visuals *(int, int)
        """

    def setShadowOrtho(self, halfSize: float, zNear: float, zFar: float) -> None:
        """
        /**
           * @brief Set the directional shadow box (orthographic) parameters.
           *
           * Defaults are 12.5, 0.1 and 55 m; overwritten by setRenderQualitySettings().
           * @param halfSize Half-size of the ortho box in metres.
           * @param zNear Near plane in metres.
           * @param zFar Far plane in metres.
           */
        C++: void (float, float, float)
        """

    def setShadowCenterOffset(self, meters: float) -> None:
        """
        /**
           * @brief Set the shadow center offset distance along the camera forward vector.
           *
           * Default 10 m; overwritten by setRenderQualitySettings().
           * @param meters Distance ahead of the camera.
           */
        C++: void (float)
        """

    def requestShadowUpdate(self) -> None:
        """
        /**
           * @brief Mark viewer shadow maps stale so they are rebuilt on the next viewer frame.
           *
           * This is useful when RenderQualitySettings::updateShadowsEveryFrame is false and
           * the application moves static-looking lights or scene geometry manually.
           */
        C++: void ()
        """

    def setUpdateShadowsEveryFrame(self, enabled: bool) -> None:
        """
        /**
           * @brief Runtime shortcut for RenderQualitySettings::updateShadowsEveryFrame.
           * @param enabled True to refresh shadow maps every viewer frame, false to update on demand.
           */
        C++: void (bool)
        """

    def updateShadowsEveryFrame(self) -> bool:
        """
        /**
           * @brief True when viewer shadows are refreshed every frame.
           * @return RenderQualitySettings::updateShadowsEveryFrame.
           */
        C++: bool () const
        """

    def getShadowCenterOffset(self) -> float:
        """
        /**
           * @brief Get the current shadow center offset distance.
           * @return Distance ahead of the camera.
           */
        C++: float () const
        """

    def captureSupersampledRgba(self, externalCam: Camera, scale: int, ov: RayraiWindow.RenderOverrides) -> RayraiWindow.SupersampledCapture:
        """
        /**
           * @brief Render an explicit supersampled viewer capture and downsample to the input camera size.
           *
           * This is intended for documentation screenshots. It is opt-in and uses an internal temporary
           * camera, so it does not resize or otherwise mutate @p externalCam. Temporal AA and viewer
           * upscaling are disabled for the capture.
           *
           * @param externalCam Camera whose pose, projection and render-target size define the output.
           * @param scale Supersampling factor per axis, clamped to [1, 4].
           * @param ov Render overrides, as for renderWithExternalCamera().
           * @return Downsampled RGBA8 image at the camera's render-target size.
           */
        C++: SupersampledCapture (const Camera &, int, const RenderOverrides &)
        """

    @overload
    def renderWithExternalCamera(self, sensorCam: raisimpy.RGBCamera, externalCam: Camera, ov: RayraiWindow.RenderOverrides) -> None:
        """
        /**
           * @brief Render using a raisim RGB camera and an external render camera.
           *
           * Only the sensor's world pose is copied into @p externalCam (see
           * syncRaisimCameraPose(): sensor +X is the view direction, +Z is up);
           * projection and resolution come from @p externalCam itself. Custom visuals,
           * point clouds and coordinate frames are never drawn, MSAA/temporal AA/viewer
           * upscaling are disabled, and weather effects follow
           * WeatherSettings::affectSensors, regardless of @p ov.
           * @param sensorCam Raisim RGB camera providing the pose; its pose is updated.
           * @param externalCam Render camera to use for rendering; receives the output.
           * @param ov Shader/toggle overrides (see RenderOverrides).
           */
        C++: void (raisim::RGBCamera &, Camera &, const RenderOverrides &)
        """

    @overload
    def renderWithExternalCamera(self, sensorCam: raisimpy.DepthCamera, externalCam: Camera, ov: RayraiWindow.RenderOverrides) -> None:
        """
        /**
           * @brief Render using a raisim depth camera's pose and an external render camera.
           *
           * Same as the RGB overload (pose only, sensor-safe overrides). This renders
           * the color image; use renderDepthPlaneDistance() for linear depth.
           * @param sensorCam Raisim depth camera providing the pose; its pose is updated.
           * @param externalCam Render camera to use for rendering; receives the output.
           * @param ov Shader/toggle overrides (see RenderOverrides).
           */
        C++: void (raisim::DepthCamera &, Camera &, const RenderOverrides &)
        """

    @overload
    def renderWithExternalCamera(self, externalCam: Camera, ov: RayraiWindow.RenderOverrides) -> None:
        """
        /**
           * @brief Render using an external render camera.
           *
           * Renders into @p externalCam's own render targets (allocated on first use at
           * its configured size); read the result from the camera, e.g. its final
           * texture. Syncs raisim objects and finalizes at most one pending async mesh
           * import first. When nothing that affects the image has changed since the last
           * call with the same camera, the previous frame may be reused; call
           * invalidateExternalFrameCache() after changes the renderer cannot see.
           * Requires this renderer's GL context.
           *
           * @param externalCam Render camera to use for rendering; receives the output.
           * @param ov Shader/toggle overrides (see RenderOverrides). Throws
           *        std::invalid_argument when a custom @c post shader is combined with
           *        reprojected temporal AA.
           */
        C++: void (Camera &, const RenderOverrides &)
        """

    def buildExternalFrameCacheKey(self, externalCam: Camera, ov: RayraiWindow.RenderOverrides, useTemporalAa: bool, sceneMsaaSamples: int, pointCloudsMustBeDetectable: bool = False) -> tuple:
        """
        /**
           * @brief Internal: compute the frame-cache key for an external render.
           * @param externalCam Camera to be rendered.
           * @param ov Overrides to be used.
           * @param useTemporalAa Whether temporal AA will run.
           * @param sceneMsaaSamples Scene MSAA sample count.
           * @param[out] key Resulting key.
           * @param pointCloudsMustBeDetectable Only detectable point clouds contribute.
           * @return False when the frame must not be cached.
           */
        C++: bool (const Camera &, const RenderOverrides &, bool, int, uint64_t &, bool) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def captureRenderPassTimings(self, externalCam: Camera, ov: RayraiWindow.RenderOverrides) -> list[RayraiWindow.RenderPassTiming]:
        """
        /**
           * @brief Explicitly render once and return GPU elapsed-time measurements by pass.
           *
           * This inserts blocking OpenGL timer queries and is intended for diagnostics and
           * benchmarks only. Normal rendering does not issue timer queries. Viewer upscaling
           * is disabled for the measurement; scenes with foliage or water are rendered again
           * to time their sub-passes (`water_*` entries for water).
           *
           * @param externalCam Camera to render with (receives the image).
           * @param ov Render overrides.
           * @return One entry per timed pass, in submission order.
           */
        C++: std::vector<RenderPassTiming> (Camera &, const RenderOverrides &)
        """

    def captureViewerPassTimings(self, width: int, height: int) -> list[RayraiWindow.RenderPassTiming]:
        """
        /** @brief Blocking diagnostic renders of the native viewer, including non-detectable
           * foliage. Call with a current GL context; timings exclude presentation/UI.
           * Runs update() on the internal camera, again for foliage and for water sub-passes.
           * @param width Viewer width in pixels; must be positive (throws std::invalid_argument).
           * @param height Viewer height in pixels; must be positive.
           * @return One entry per timed pass. */
        C++: std::vector<RenderPassTiming> (int, int)
        """

    def conservativeOcclusionCullingDiagnostics(self) -> RayraiWindow.ConservativeOcclusionCullingDiagnostics:
        """
        /// @return Diagnostics: occlusion-culling counters of the last scene color pass.
        C++: ConservativeOcclusionCullingDiagnostics () const
        """

    def foliageRenderDiagnostics(self) -> RayraiWindow.FoliageRenderDiagnostics:
        """
        /// @return Diagnostics: foliage/instanced draw counters of the last rendered frame.
        C++: FoliageRenderDiagnostics () const
        """

    def environmentBackgroundDiagnostics(self) -> RayraiWindow.EnvironmentBackgroundDiagnostics:
        """
        /// @return Diagnostics: how the last scene pass drew the environment background.
        C++: EnvironmentBackgroundDiagnostics () const
        """

    def shadowCacheDiagnostics(self) -> RayraiWindow.ShadowCacheDiagnostics:
        """
        /// @return Diagnostics: shadow-map caching counters of the last rendered frame.
        C++: ShadowCacheDiagnostics () const
        """

    def captureDebugPasses(self, externalCam: Camera, ov: RayraiWindow.RenderOverrides) -> list[RayraiWindow.DebugPassCapture]:
        """
        /**
           * @brief Capture the standard RayRai PBR/postprocess debug pass set for any scene.
           *
           * The capture uses the caller-provided camera and render overrides, temporarily switches to
           * Ultra quality, disables temporal history/upscaling for deterministic per-pass readback, and
           * afterwards re-applies the previous quality settings through setRenderQualitySettings() and
           * resets the renderer-local PBR debug mode to -1 (process default). Because it goes through
           * setRenderQualitySettings(), application-added additional lights are cleared and the preset
           * reads Custom afterwards (see setRenderQualitySettings()). Returned pixels are
           * top-left ordered RGBA8 buffers so callers can write PNGs or analyze them without a RayRai image
           * writer dependency.
           *
           * @param externalCam Camera to render with; its size defines the capture size.
           * @param ov Render overrides (temporal AA and upscaling are forced off).
           * @return 25 passes (final, material channels, lighting terms, shadows, fog, AO, ...).
           */
        C++: std::vector<DebugPassCapture> (Camera &, const RenderOverrides &)
        """

    def planDirectionalShadowCascades(self, camera: Camera) -> RayraiWindow.DirectionalShadowCascadePlan:
        """
        /**
           * @brief Compute stable directional-light cascade splits and light-space bounds.
           *
           * The shadow pass uses this same planner for the main directional light; calling it is
           * CPU-only and does not allocate shadow maps or change renderer state. Cascades are enabled
           * when RenderQualitySettings::directionalShadowCascadeCount is greater than 1.
           *
           * @param camera Camera whose frustum is split.
           * @return Cascade plan (see DirectionalShadowCascadePlan).
           */
        C++: DirectionalShadowCascadePlan (const Camera &) const
        """

    def planAdditionalShadowAtlas(self) -> RayraiWindow.ShadowAtlasPlan:
        """
        /**
           * @brief Compute a deterministic tile plan for selected local-light shadow maps.
           *
           * The renderer uses this plan for its 2D additional-light shadow atlas (point lights keep
           * standalone cubemaps unless the point-shadow policy says otherwise). Calling it is CPU-only;
           * the result depends on the shadow lights selected by the last rendered frame.
           *
           * @return Atlas plan (see ShadowAtlasPlan).
           */
        C++: ShadowAtlasPlan () const
        """

    def shadowDebugOverlaySummary(self) -> RayraiWindow.ShadowDebugOverlaySummary:
        """
        /**
           * @brief Build the CPU summary used by the optional in-view shadow debug overlay.
           *
           * This does not render UI, allocate GL resources, or mutate settings. It is useful for tests,
           * benchmarks, and applications that want to show their own shadow tuning panel.
           * @return Summary for the internal camera.
           */
        C++: ShadowDebugOverlaySummary () const
        """

    @staticmethod
    def reflectionProbeCaptureFrame(face: int) -> RayraiWindow.ReflectionProbeCaptureFrame:
        """
        /**
           * @brief Return the capture camera frame used for a cubemap face.
           *
           * The frame follows the same OpenGL cubemap convention used by environment-map sampling and
           * CPU probe filtering. It is exposed so tests and tools can validate local reflection probe
           * orientation without rendering a probe.
           *
           * @param face Face index in GL order (+X, -X, +Y, -Y, +Z, -Z), clamped to [0, 5].
           * @return The face's camera basis.
           */
        C++: ReflectionProbeCaptureFrame (int)
        """

    @overload
    def captureReflectionProbeCubemap(self, position: numpy.ndarray) -> int:
        """
        /**
           * @brief Capture a static local reflection probe into a cubemap.
           *
           * This is a raster, load-time/manual operation. It renders six 90-degree faces
           * from @p position and returns an RGBA8 OpenGL cubemap id with mipmaps. The caller owns
           * the returned texture and may bind it to PBR materials through Visuals or Material.
           * Faces are rendered without post-processing. Requires this renderer's GL context.
           *
           * @param position World-space capture position.
           * @return OpenGL cubemap id, or 0 on failure.
           */
        C++: unsigned int (const glm::vec3 &)
        """

    @overload
    def captureReflectionProbeCubemap(self, position: numpy.ndarray, settings: RayraiWindow.ReflectionProbeCaptureSettings) -> int:
        """
        /**
           * @brief Capture a static local reflection probe with explicit capture settings.
           * @param position World-space capture position.
           * @param settings What to draw and at which resolution.
           * @return Caller-owned OpenGL cubemap id, or 0 on failure.
           */
        C++: unsigned int (const glm::vec3 &, const ReflectionProbeCaptureSettings &)
        """

    @overload
    def captureFilteredReflectionProbe(self, position: numpy.ndarray, radius: float, strength: float) -> RayraiWindow.ReflectionProbe:
        """
        /**
           * @brief Capture a static local probe and generate irradiance, prefiltered specular, and BRDF maps.
           *
           * This is an explicit bounded setup operation for High/Ultra. It does not update automatically
           * after scene changes and does not affect Fast unless the caller uses the returned probe.
           * Uses default capture settings and reflectionProbeFilterSettingsForCurrentQuality(), so with
           * probe filtering disabled only the raw environment map is produced. All returned textures are
           * owned by the caller. Pass the probe to addReflectionProbe() to make it selectable.
           *
           * @param position World-space capture position.
           * @param radius Influence radius in metres (negative values become 0 = unbounded).
           * @param strength Probe strength (negative values become 0).
           * @return The probe; map ids are 0 for maps that were skipped or failed.
           */
        C++: ReflectionProbe (const glm::vec3 &, float, float)
        """

    @overload
    def captureFilteredReflectionProbe(self, position: numpy.ndarray, radius: float, strength: float, captureSettings: RayraiWindow.ReflectionProbeCaptureSettings) -> RayraiWindow.ReflectionProbe:
        """
        /// @copybrief captureFilteredReflectionProbe(const glm::vec3&,float,float)
          /// Uses filter settings for the current quality.
          /// @param position World-space capture position.
          /// @param radius Influence radius in metres.
          /// @param strength Probe strength.
          /// @param captureSettings What to draw and at which resolution.
          /// @return The probe; textures are owned by the caller.
        C++: ReflectionProbe (const glm::vec3 &, float, float, const ReflectionProbeCaptureSettings &)
        """

    @overload
    def captureFilteredReflectionProbe(self, position: numpy.ndarray, radius: float, strength: float, captureSettings: RayraiWindow.ReflectionProbeCaptureSettings, filterSettings: RayraiWindow.ReflectionProbeFilterSettings) -> RayraiWindow.ReflectionProbe:
        """
        /// @copybrief captureFilteredReflectionProbe(const glm::vec3&,float,float)
          /// @param position World-space capture position.
          /// @param radius Influence radius in metres.
          /// @param strength Probe strength.
          /// @param captureSettings What to draw and at which resolution.
          /// @param filterSettings CPU filtering settings; zero sizes skip the corresponding map.
          /// @return The probe; textures are owned by the caller.
        C++: ReflectionProbe (const glm::vec3 &, float, float, const ReflectionProbeCaptureSettings &, const ReflectionProbeFilterSettings &)
        """

    @overload
    def captureFilteredReflectionProbe(self, opts: RayraiWindow.CaptureFilteredReflectionProbeOptions) -> RayraiWindow.ReflectionProbe:
        """
        /// @brief Options-struct overload; prefer this in new code.
          /// @param opts Capture options.
          /// @return The probe; textures are owned by the caller.
        C++: ReflectionProbe (const CaptureFilteredReflectionProbeOptions &)
        """

    @overload
    def captureFilteredReflectionProbeCached(self, opts: RayraiWindow.CaptureFilteredReflectionProbeOptions) -> RayraiWindow.ReflectionProbe:
        """
        /// @brief Cached variant of the options-struct overload.
          /// @param opts Capture options.
          /// @return The cached or newly captured probe; textures are owned by the renderer.
        C++: ReflectionProbe (const CaptureFilteredReflectionProbeOptions &)
        """

    @overload
    def captureFilteredReflectionProbeCached(self, position: numpy.ndarray, radius: float, strength: float) -> RayraiWindow.ReflectionProbe:
        """
        /**
           * @brief Cached captureFilteredReflectionProbe(); textures stay owned by the renderer.
           *
           * Entries are keyed by all arguments and live until clearReflectionProbeCache().
           * @param position World-space capture position.
           * @param radius Influence radius in metres.
           * @param strength Probe strength.
           * @return The cached or newly captured probe.
           */
        C++: ReflectionProbe (const glm::vec3 &, float, float)
        """

    @overload
    def captureFilteredReflectionProbeCached(self, position: numpy.ndarray, radius: float, strength: float, captureSettings: RayraiWindow.ReflectionProbeCaptureSettings) -> RayraiWindow.ReflectionProbe:
        """
        /// @copybrief captureFilteredReflectionProbeCached(const glm::vec3&,float,float)
          /// @param position World-space capture position.
          /// @param radius Influence radius in metres.
          /// @param strength Probe strength.
          /// @param captureSettings Capture settings.
          /// @return The cached or newly captured probe.
        C++: ReflectionProbe (const glm::vec3 &, float, float, const ReflectionProbeCaptureSettings &)
        """

    @overload
    def captureFilteredReflectionProbeCached(self, position: numpy.ndarray, radius: float, strength: float, captureSettings: RayraiWindow.ReflectionProbeCaptureSettings, filterSettings: RayraiWindow.ReflectionProbeFilterSettings) -> RayraiWindow.ReflectionProbe:
        """
        /// @copybrief captureFilteredReflectionProbeCached(const glm::vec3&,float,float)
          /// @param position World-space capture position.
          /// @param radius Influence radius in metres.
          /// @param strength Probe strength.
          /// @param captureSettings Capture settings.
          /// @param filterSettings Filter settings.
          /// @return The cached or newly captured probe.
        C++: ReflectionProbe (const glm::vec3 &, float, float, const ReflectionProbeCaptureSettings &, const ReflectionProbeFilterSettings &)
        """

    def captureReflectionProbeCubemapCached(self, position: numpy.ndarray, settings: RayraiWindow.ReflectionProbeCaptureSettings) -> int:
        """
        /**
           * @brief Capture or reuse a cached static local reflection probe cubemap.
           *
           * The cache is explicit and does not recapture automatically. The returned
           * cubemap is owned by RayRai until clearReflectionProbeCache() or destruction.
           * Applications should clear the cache after changing scene content that should
           * be reflected in future probe captures. Entries are keyed by position and settings.
           *
           * @param position World-space capture position.
           * @param settings Capture settings.
           * @return Renderer-owned cubemap id, or 0 on failure (failures are not cached).
           */
        C++: unsigned int (const glm::vec3 &, const ReflectionProbeCaptureSettings &)
        """

    def clearReflectionProbeCache(self) -> None:
        """
        /// Delete every texture held by the probe caches and empty them. GL context required.
          /// Probes previously returned by the cached functions become dangling.
        C++: void ()
        """

    def reflectionProbeCacheSize(self) -> int:
        """
        /// @return Number of entries in both probe caches.
        C++: size_t () const
        """

    @staticmethod
    def loadReflectionProbeSidecar(sceneFile: str, settings: RayraiWindow.ReflectionProbeSidecarSettings | None = None) -> tuple:
        """
        /**
           * @brief Parse the reflection-probe sidecar that belongs to a scene file (CPU only).
           *
           * Sidecar lookup (shared by all sidecar loaders): if @p sceneFile itself ends in
           * `.json` it is used; otherwise `<sceneFile>.rayrai_probes.json`, then
           * `<dir>/<stem>.rayrai_probes.json`. The first existing file wins. Nothing is
           * captured; use the returned entries with captureFilteredReflectionProbe().
           *
           * @param sceneFile Scene/mesh file path, or the sidecar path itself.
           * @param[out] settings Optional `quality` block; reset to defaults first.
           * @param[out] loadedSidecarPath Optional path of the file that was read (empty if none).
           * @return Parsed probes; empty when no sidecar exists or it has no `probes` array.
           */
        C++: std::vector<ReflectionProbeSidecarProbe> (const std::string &, ReflectionProbeSidecarSettings *, std::string *)
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    @staticmethod
    def loadReflectionProbeSidecarResult(sceneFile: str) -> RayraiWindow.ReflectionProbeSidecarLoadResult:
        """
        /// @brief Struct-returning overload; prefer this in new code.
          /// @param sceneFile Scene/mesh file path, or the sidecar path itself.
          /// @return Probes, quality settings and the sidecar path.
        C++: ReflectionProbeSidecarLoadResult (const std::string &)
        """

    def applyReflectionProbeSidecarQualitySettings(self, sceneFile: str) -> bool:
        """
        /**
           * @brief Apply the `quality` block of a scene's reflection-probe sidecar.
           *
           * Copies the provided probe selection/filter values into the current render
           * quality settings via setRenderQualitySettings() (which has side effects, see
           * there). Probes themselves are not captured.
           * @param sceneFile Scene/mesh file path, or the sidecar path itself.
           * @return True when the sidecar provided at least one quality value.
           */
        C++: bool (const std::string &)
        """

    @staticmethod
    def loadEnvironmentSidecar(sceneFile: str) -> tuple:
        """
        /**
           * @brief Parse a scene's `.rayrai_environment.json` sidecar (CPU only).
           *
           * Uses the same lookup as loadReflectionProbeSidecar().
           * @param sceneFile Scene/mesh file path, or the sidecar path itself.
           * @param[out] loadedSidecarPath Optional path of the file that was read (empty if none).
           * @return Parsed values with presence flags; all flags false when no sidecar exists.
           */
        C++: EnvironmentSidecarSettings (const std::string &, std::string *)
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def applyEnvironmentSidecarSettings(self, sceneFile: str) -> bool:
        """
        /**
           * @brief Apply a scene's environment sidecar to the current render quality settings.
           *
           * Present values are written into a copy of the current settings, which is
           * then applied with setRenderQualitySettings() (see its side effects). A
           * background map is loaded as a 512 px cubemap and installed with
           * setEnvironmentBackground(). Requires this renderer's GL context.
           * @param sceneFile Scene/mesh file path, or the sidecar path itself.
           * @return True when the sidecar provided at least one value.
           */
        C++: bool (const std::string &)
        """

    @staticmethod
    def loadWeatherSidecar(sceneFile: str) -> tuple:
        """
        /**
           * @brief Parse a scene's `.rayrai_weather.json` sidecar (CPU only).
           *
           * Uses the same lookup as loadReflectionProbeSidecar(). Weather keys are read
           * from a `weather` object when present, otherwise from the top level.
           * @param sceneFile Scene/mesh file path, or the sidecar path itself.
           * @param[out] loadedSidecarPath Optional path of the file, set only when it contained weather or fog data.
           * @return Parsed settings; both presence flags false when nothing was found.
           */
        C++: WeatherSidecarSettings (const std::string &, std::string *)
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def applyWeatherSidecarSettings(self, sceneFile: str) -> bool:
        """
        /**
           * @brief Apply a scene's weather sidecar.
           *
           * Calls setWeatherSettings() when weather data is present, and replaces all
           * local fog volumes when the sidecar has `local_fog_volumes`.
           * @param sceneFile Scene/mesh file path, or the sidecar path itself.
           * @return True when anything was applied.
           */
        C++: bool (const std::string &)
        """

    @staticmethod
    def writeWeatherSidecar(sidecarFile: str, settings: WeatherSettings, localFogVolumes: Sequence[LocalFogVolume] = []) -> bool:
        """
        /**
           * @brief Write weather settings (and optional fog volumes) as a weather sidecar JSON file.
           * @param sidecarFile Output path; overwritten.
           * @param settings Weather settings to write.
           * @param localFogVolumes Fog volumes to write; at most kMaxLocalFogVolumes are written.
           * @return False when the file cannot be opened for writing.
           */
        C++: bool (const std::string &, const WeatherSettings &, const std::vector<LocalFogVolume> &)
        """

    def reflectionProbeFilterSettingsForCurrentQuality(self) -> RayraiWindow.ReflectionProbeFilterSettings:
        """
        /// @return Probe filter settings derived from the current RenderQualitySettings
          /// (all zero when reflection-probe filtering is disabled).
        C++: ReflectionProbeFilterSettings () const
        """

    def addReflectionProbe(self, probe: RayraiWindow.ReflectionProbe) -> int:
        """
        /**
           * @brief Register a captured probe for selectReflectionProbeBlend() / applyNearestReflectionProbe().
           *
           * The probe is copied with radius/strength clamped to >= 0 and box projection
           * disabled for degenerate boxes. Registered probes are not bound to visuals
           * automatically, and the registry does not take ownership of the textures.
           * @param probe Probe to register; ignored when it has no environment, irradiance or prefiltered map.
           * @return Number of registered probes afterwards.
           */
        C++: size_t (const ReflectionProbe &)
        """

    def clearReflectionProbes(self) -> None:
        """
        /// Unregister every probe added with addReflectionProbe(); their textures stay with the caller.
        C++: void ()
        """

    def selectReflectionProbeBlend(self, position: numpy.ndarray, blend: RayraiWindow.ReflectionProbeBlend) -> bool:
        """
        /**
           * @brief Pick up to two registered probes influencing a position.
           *
           * Weight per probe is `strength * (1 - d / radius)` inside a positive radius
           * (0 outside), or `strength / (1 + d)` for radius 0, where d is the distance to
           * the probe position, or to its box when box projection is valid. Probes at or
           * below RenderQualitySettings::minReflectionProbeWeight are skipped; the
           * second probe is used only when maxReflectionProbesPerSelection is 2.
           * @param position World-space query position.
           * @param[out] blend Selected probes and normalized weights.
           * @return True when at least one probe influences the position.
           */
        C++: bool (const glm::vec3 &, ReflectionProbeBlend &) const
        """

    def reflectionProbeDebugView(self, position: numpy.ndarray) -> RayraiWindow.ReflectionProbeDebugView:
        """
        /**
           * @brief Diagnostics: probe selection at a position plus per-probe weights.
           * @param position World-space query position.
           * @return Debug view; CPU only.
           */
        C++: ReflectionProbeDebugView (const glm::vec3 &) const
        """

    def reflectionProbeDebugOverlaySummary(self, position: numpy.ndarray) -> RayraiWindow.ReflectionProbeDebugOverlaySummary:
        """
        /**
           * @brief Build the CPU summary used by the optional in-view reflection probe overlay.
           *
           * This is diagnostics-only: it does not capture probes, filter maps, allocate GL resources,
           * or mutate probe selection state.
           * @param position World-space query position.
           * @return Overlay summary.
           */
        C++: ReflectionProbeDebugOverlaySummary (const glm::vec3 &) const
        """

    def suggestReflectionProbePlacementFromSceneBounds(self) -> RayraiWindow.ReflectionProbePlacementSuggestion:
        """
        /**
           * @brief Suggest a static local reflection-probe placement from current scene bounds.
           *
           * This is CPU-only and non-mutating. It does not capture or register a probe; callers can use the
           * returned position/radius/box bounds with captureFilteredReflectionProbe() or a sidecar file after
           * explicit approval. Halfspaces and heightmaps are ignored so infinite/default grounds do not
           * dominate imported-room probe placement. Bounds are the union of approximate bounding
           * spheres of raisim objects, custom visuals and instanced visuals.
           * @return Suggestion; `valid` is false when there is nothing to bound.
           */
        C++: ReflectionProbePlacementSuggestion () const
        """

    def addLocalFogVolume(self, volume: LocalFogVolume) -> int:
        """
        /** @name Local fog volumes, projected decals and irradiance volumes
           *  Scene-effect lists rendered by the PBR path. Each `add*()` copies and
           *  sanitizes its argument (e.g. clamps strengths and fades, orthonormalizes
           *  box orientations), silently ignores it once the matching `kMax*` limit is
           *  reached, and returns the list size afterwards. The lists are cleared by
           *  swapWorld(). See SceneEffects.hpp / Weather.hpp for the field meanings. */
          ///@{
          /// Add a spherical local fog volume (density clamped to [-1, 1]; negative values clear fog).
          /// @param volume Volume to add. @return Number of volumes afterwards (capped at kMaxLocalFogVolumes).
        C++: size_t (const LocalFogVolume &)
        """

    def clearLocalFogVolumes(self) -> None:
        """
        /// Remove all local fog volumes.
        C++: void ()
        """

    def localFogVolumeCount(self) -> int:
        """
        /// @return Number of local fog volumes.
        C++: size_t () const
        """

    def localFogVolumes(self) -> list[LocalFogVolume]:
        """
        /// @return Current local fog volumes (sanitized copies).
        C++: const std::vector<LocalFogVolume> &() const
        """

    def addProjectedDecal(self, decal: ProjectedDecal) -> int:
        """
        /// Add a projected (box) decal.
          /// @param decal Decal to add. @return Number of decals afterwards (capped at kMaxProjectedDecals).
        C++: size_t (const ProjectedDecal &)
        """

    def clearProjectedDecals(self) -> None:
        """
        /// Remove all projected decals.
        C++: void ()
        """

    def projectedDecalCount(self) -> int:
        """
        /// @return Number of projected decals.
        C++: size_t () const
        """

    def projectedDecals(self) -> list[ProjectedDecal]:
        """
        /// @return Current projected decals (sanitized copies).
        C++: const std::vector<ProjectedDecal> &() const
        """

    def addIrradianceVolume(self, volume: IrradianceVolume) -> int:
        """
        /// Add a box-shaped irradiance volume.
          /// @param volume Volume to add. @return Number of volumes afterwards (capped at kMaxIrradianceVolumes).
        C++: size_t (const IrradianceVolume &)
        """

    def clearIrradianceVolumes(self) -> None:
        """
        /// Remove all irradiance volumes.
        C++: void ()
        """

    def irradianceVolumeCount(self) -> int:
        """
        /// @return Number of irradiance volumes.
        C++: size_t () const
        """

    def irradianceVolumes(self) -> list[IrradianceVolume]:
        """
        /// @return Current irradiance volumes (sanitized copies).
        C++: const std::vector<IrradianceVolume> &() const
        """

    def setWaterBody(self, name: str, body: WaterBody) -> None:
        """
        ///@}
          /** @name Water bodies
           *  Oceans and rectangular pools drawn by every color render (see Water.hpp). The surface
           *  moves with the water time, which only changes through setWaterTimeSeconds() and
           *  advanceWaterTime(), so renders are repeatable. Water bodies are cleared by swapWorld().
           *  Quality controls live in RenderQualitySettings (`water*`). */
          ///@{
          /**
           * @brief Add a water body, or replace the one with the same name.
           * @param name Name of the body.
           * @param body Body to copy; wave directions are normalized.
           * @throws std::invalid_argument if @p name is empty or a field is out of range (validateWaterBody()).
           */
        C++: void (const std::string &, const WaterBody &)
        """

    def removeWaterBody(self, name: str) -> bool:
        """
        /// Remove a water body. @param name Name of the body. @return False if there was none.
        C++: bool (const std::string &)
        """

    def clearWaterBodies(self) -> None:
        """
        /// Remove all water bodies.
        C++: void ()
        """

    def findWaterBody(self, name: str) -> WaterBody | None:
        """
        /// @param name Name of the body. @return The body, or nullptr if there is none.
        C++: const WaterBody *(const std::string &) const
        """

    def waterBodyCount(self) -> int:
        """
        /// @return Number of water bodies.
        C++: size_t () const
        """

    def waterBodyNames(self) -> list[str]:
        """
        /// @return Names of the water bodies in the order they were added.
        C++: std::vector<std::string> () const
        """

    def setWaterTimeSeconds(self, seconds: float) -> None:
        """
        /// Set the water time (finite; non-finite values are ignored). @param seconds Time, seconds.
        C++: void (double)
        """

    def advanceWaterTime(self, deltaSeconds: float) -> None:
        """
        /// Advance the water time. @param deltaSeconds Step, seconds (finite).
        C++: void (double)
        """

    def waterTimeSeconds(self) -> float:
        """
        /// @return Current water time, seconds.
        C++: double () const
        """

    def sampleWaterSurface(self, worldXY: numpy.ndarray) -> WaterSurfaceSample:
        """
        /**
           * @brief Highest water surface above a horizontal world position at the current water time.
           * @param worldXY Horizontal world position, metres.
           * @return Sample of the highest body containing the position; `inside` is false if none does.
           */
        C++: WaterSurfaceSample (const glm::dvec2 &) const
        """

    def sampleWaterFlow(self, worldXY: numpy.ndarray) -> WaterFlowSample:
        """
        /**
           * @brief Simulated water above a horizontal world position (bodies with WaterFlow::enabled).
           *
           * The simulation runs on the GPU when a color render finds the water time advanced. This reads
           * the state copied back after an earlier render, so it trails the drawn water by a render or
           * two (WaterFlowSample::timeSeconds). Where a moving body covers the position it reports the
           * water around that body (see WaterFlowSample).
           * @param worldXY Horizontal world position, metres.
           * @return Sample of the highest simulated surface over the position; `inside` is false if no
           *         simulated body holds it or none has been read back yet.
           */
        C++: WaterFlowSample (const glm::dvec2 &) const
        """

    def resetWaterFlow(self, name: str) -> bool:
        """
        /**
           * @brief Start a simulated body over: capture its bed again and fill it with still water up to
           * its level at the next render.
           * @param name Name of the body. @return False if there is no simulated body of that name.
           */
        C++: bool (const std::string &)
        """

    def waterDiagnostics(self) -> WaterDiagnostics:
        """
        /// @return Water renderer state of the last color render of the scene.
        C++: WaterDiagnostics () const
        """

    @staticmethod
    def defaultWeatherSettings(preset: WeatherPreset) -> WeatherSettings:
        """
        ///@}
          /** @name Weather
           *  While WeatherSettings::enabled is true, the resolved weather is applied on
           *  top of the last settings passed to setRenderQualitySettings(): main light,
           *  ambient, sky, clouds and wet/snow materials are overwritten and fog is
           *  combined with the user values (see raisin::WeatherSettings). Disabling
           *  weather restores the user settings; getUserRenderQualitySettings() returns
           *  them meanwhile. Transitions and wetness advance only in updateWeather(). */
          ///@{
          /**
           * @brief Default settings for a weather preset (CPU only).
           * @param preset Preset to expand.
           * @return Settings with `enabled = true`, High quality, 13:00 local time.
           */
        C++: WeatherSettings (WeatherPreset)
        """

    def setWeatherSettings(self, settings: WeatherSettings) -> None:
        """
        /**
           * @brief Apply weather settings immediately.
           *
           * Values are clamped to their valid ranges, any running transition is
           * cancelled, a generated weather sky environment is released, and the
           * render quality settings are updated (see the group description).
           * @param settings New weather settings.
           */
        C++: void (const WeatherSettings &)
        """

    def getWeatherSettings(self) -> WeatherSettings:
        """
        /// @return Current (clamped) weather settings; mid-transition this is the blended state.
        C++: const WeatherSettings &() const
        """

    def setWeatherPreset(self, preset: WeatherPreset) -> None:
        """
        /// Apply defaultWeatherSettings(@p preset) immediately.
          /// @param preset Preset to apply.
        C++: void (WeatherPreset)
        """

    @overload
    def transitionWeather(self, preset: WeatherPreset, transitionSeconds: float) -> None:
        """
        /// Blend from the current weather to a preset's defaults over time.
          /// @param preset Target preset.
          /// @param transitionSeconds Duration; <= 0 applies the target immediately.
        C++: void (WeatherPreset, double)
        """

    @overload
    def transitionWeather(self, target: WeatherSettings, transitionSeconds: float) -> None:
        """
        /**
           * @brief Blend from the current weather to @p target over time.
           *
           * The blend advances only in updateWeather().
           * @param target Target settings.
           * @param transitionSeconds Duration in seconds; <= 0 applies the target immediately.
           */
        C++: void (const WeatherSettings &, double)
        """

    def updateWeather(self, dt: float) -> None:
        """
        /**
           * @brief Advance weather transitions, wetness accumulation/drying and lightning.
           *
           * Call once per frame from the render thread before rendering.
           * @param dt Elapsed time in seconds (negative values are treated as 0).
           */
        C++: void (double)
        """

    def weatherDiagnostics(self) -> WeatherDiagnostics:
        """
        /// Resolved weather state (sun/moon, clouds, fog, precipitation, wetness, lightning). CPU only.
          /// @return Diagnostics for the current settings.
        C++: WeatherDiagnostics () const
        """

    def printWeatherDiagnostics(self) -> tuple:
        """
        /// Print weatherDiagnostics() in a human-readable multi-line form.
          /// @param out Output stream.
        C++: void (std::ostream &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def generateWeatherSkyEnvironment(self, environmentFaceSize: int = 256, irradianceFaceSize: int = 32, setVisibleBackground: bool = True) -> WeatherSkyEnvironment:
        """
        /**
           * @brief Generate a weather-driven sky/environment cubemap on demand.
           *
           * This is intentionally explicit. Applications should call it at weather or time-of-day
           * transition points, not every frame. When setVisibleBackground is true, the generated radiance
           * cubemap is also installed as the visible environment background. The previous generated
           * environment is released first; the textures are owned by the renderer and are also released by
           * clearWeatherSkyEnvironment() and setWeatherSettings(). Requires this renderer's GL context.
           *
           * @param environmentFaceSize Radiance cubemap face size, clamped to [16, 2048].
           * @param irradianceFaceSize Irradiance cubemap face size, clamped to [0, 256]; 0 skips it.
           * @param setVisibleBackground Install the radiance map as the visible background.
           * @return The generated environment (also available via weatherSkyEnvironment()).
           */
        C++: WeatherSkyEnvironment (int, int, bool)
        """

    def clearWeatherSkyEnvironment(self) -> None:
        """
        /// Delete the generated weather sky textures; unsets the visible background if it was installed.
        C++: void ()
        """

    def weatherSkyEnvironment(self) -> WeatherSkyEnvironment:
        """
        /// @return Last generated weather sky environment (`valid` false when none).
        C++: const WeatherSkyEnvironment &() const
        """

    def applyNearestReflectionProbe(self, visual: Visuals, position: numpy.ndarray) -> bool:
        """
        ///@}
          /**
           * @brief Bind the best registered reflection probe at @p position to a visual's PBR environment.
           *
           * Uses the primary probe of selectReflectionProbeBlend() with the blend strength.
           * @param visual Visual to update.
           * @param position World-space position used for selection (typically the visual's).
           * @return False when no probe influences the position (the visual is left unchanged).
           */
        C++: bool (Visuals &, const glm::vec3 &) const
        """

    def reflectionProbeCount(self) -> int:
        """
        /// @return Number of probes registered with addReflectionProbe().
        C++: size_t () const
        """

    def pickWithExternalCamera(self, externalCam: Camera, x: int, y: int, pickingOverride: Shader | None = None) -> int:
        """
        /**
           * @brief Run the picking pass and return object id at a pixel.
           * @param externalCam Render camera used for the picking pass.
           * @param x Pixel x coordinate in framebuffer space (top-left origin).
           * @param y Pixel y coordinate in framebuffer space, growing downward.
           * @param pickingOverride Optional picking shader override.
           * @return Encoded object id (24-bit), or 0 if nothing is picked or the pixel is out of range.
           */
        C++: uint32_t (Camera &, int, int, Shader *)
        """

    def addPointCloud(self, name: str) -> PointCloud:
        """
        /**
           * @brief Create a named point cloud.
           * @param name Unique name for the point cloud; aborts (`RSFATAL`) if already used.
           * @return Shared pointer to the point cloud; the renderer keeps its own reference
           *         until removePointCloud() / clearVisualizationObjects() / swapWorld().
           */
        C++: std::shared_ptr<PointCloud> (const std::string &)
        """

    def removePointCloud(self, name: str) -> None:
        """
        /**
           * @brief Remove a point cloud by name.
           * @param name Name of the point cloud to remove; aborts (`RSFATAL`) if unknown.
           */
        C++: void (const std::string &)
        """

    def clearVisualizationObjects(self) -> None:
        """
        /**
           * @brief Remove all visual objects, instanced visuals, and point clouds.
           *
           * Also drops automatically maintained tendon visuals and the target visual.
           * Coordinate frames are kept.
           */
        C++: void ()
        """

    def addCoordinateFrame(self, name: str) -> CoordinateFrame:
        """
        /**
           * @brief Create a named coordinate frame set.
           * @param name Unique name for the coordinate frame set; aborts (`RSFATAL`) if already used.
           * @return Shared pointer to the coordinate frame set (also held by the renderer).
           */
        C++: std::shared_ptr<CoordinateFrame> (const std::string &)
        """

    def removeCoordinateFrame(self, name: str) -> None:
        """
        /**
           * @brief Remove a coordinate frame set by name.
           * @param name Name of the coordinate frame set to remove; aborts (`RSFATAL`) if unknown.
           */
        C++: void (const std::string &)
        """

    @overload
    def addVisualSphere(self, name: str, radius: float, colorR: float = 1.0, colorG: float = 1.0, colorB: float = 1.0, colorA: float = 1.0) -> Visuals:
        """
        /** @name Custom visuals
           *  Render-only shapes that are not part of raisim::World. Names share one
           *  namespace (addVisual*(), importVisualScene(), addCameraFrustum()) and must
           *  be unique (`RSFATAL` otherwise). Sizes are in metres, colors are RGBA in
           *  [0, 1]. A new visual sits at the world origin with identity orientation;
           *  move it through the returned Visuals. The renderer keeps its own
           *  reference until removeVisualObject(), clearVisualizationObjects() or
           *  swapWorld(). */
          ///@{
          /**
           * @brief Add a visual sphere (not part of raisim::World).
           * @param name Visual name.
           * @param radius Sphere radius.
           * @param colorR Red channel.
           * @param colorG Green channel.
           * @param colorB Blue channel.
           * @param colorA Alpha channel.
           * @return Shared pointer to the created visual.
           */
        C++: std::shared_ptr<Visuals> (const std::string &, double, float, float, float, float)
        """

    @overload
    def addVisualSphere(self, name: str, radius: float, color: numpy.ndarray) -> Visuals:
        """
        /// @brief glm::vec4 color overload; prefer this in new code.
          /// @param name Visual name. @param radius Sphere radius. @param color RGBA color.
          /// @return Shared pointer to the created visual.
        C++: std::shared_ptr<Visuals> (const std::string &, double, const glm::vec4 &)
        """

    @overload
    def addVisualBox(self, name: str, xLength: float, yLength: float, zLength: float, colorR: float = 1.0, colorG: float = 1.0, colorB: float = 1.0, colorA: float = 1.0) -> Visuals:
        """
        /**
           * @brief Add a visual box (not part of raisim::World).
           * @param name Visual name.
           * @param xLength Box size in X.
           * @param yLength Box size in Y.
           * @param zLength Box size in Z.
           * @param colorR Red channel.
           * @param colorG Green channel.
           * @param colorB Blue channel.
           * @param colorA Alpha channel.
           * @return Shared pointer to the created visual.
           */
        C++: std::shared_ptr<Visuals> (const std::string &, double, double, double, float, float, float, float)
        """

    @overload
    def addVisualBox(self, name: str, xLength: float, yLength: float, zLength: float, color: numpy.ndarray) -> Visuals:
        """
        /// @brief glm::vec4 color overload; prefer this in new code.
          /// @param name Visual name. @param xLength Size in X. @param yLength Size in Y.
          /// @param zLength Size in Z. @param color RGBA color.
          /// @return Shared pointer to the created visual.
        C++: std::shared_ptr<Visuals> (const std::string &, double, double, double, const glm::vec4 &)
        """

    @overload
    def addVisualCylinder(self, name: str, radius: float, height: float, colorR: float = 1.0, colorG: float = 1.0, colorB: float = 1.0, colorA: float = 1.0) -> Visuals:
        """
        /**
           * @brief Add a visual cylinder (not part of raisim::World).
           * @param name Visual name.
           * @param radius Cylinder radius.
           * @param height Cylinder height (along the visual's local Z axis).
           * @param colorR Red channel.
           * @param colorG Green channel.
           * @param colorB Blue channel.
           * @param colorA Alpha channel.
           * @return Shared pointer to the created visual.
           */
        C++: std::shared_ptr<Visuals> (const std::string &, double, double, float, float, float, float)
        """

    @overload
    def addVisualCylinder(self, name: str, radius: float, height: float, color: numpy.ndarray) -> Visuals:
        """
        /// @brief glm::vec4 color overload; prefer this in new code.
          /// @param name Visual name. @param radius Cylinder radius. @param height Cylinder height.
          /// @param color RGBA color.
          /// @return Shared pointer to the created visual.
        C++: std::shared_ptr<Visuals> (const std::string &, double, double, const glm::vec4 &)
        """

    @overload
    def addVisualCapsule(self, name: str, radius: float, height: float, colorR: float = 1.0, colorG: float = 1.0, colorB: float = 1.0, colorA: float = 1.0) -> Visuals:
        """
        /**
           * @brief Add a visual capsule (not part of raisim::World).
           * @param name Visual name.
           * @param radius Capsule radius.
           * @param height Capsule height.
           * @param colorR Red channel.
           * @param colorG Green channel.
           * @param colorB Blue channel.
           * @param colorA Alpha channel.
           * @return Shared pointer to the created visual.
           */
        C++: std::shared_ptr<Visuals> (const std::string &, double, double, float, float, float, float)
        """

    @overload
    def addVisualCapsule(self, name: str, radius: float, height: float, color: numpy.ndarray) -> Visuals:
        """
        /// @brief glm::vec4 color overload; prefer this in new code.
          /// @param name Visual name. @param radius Capsule radius. @param height Capsule height.
          /// @param color RGBA color.
          /// @return Shared pointer to the created visual.
        C++: std::shared_ptr<Visuals> (const std::string &, double, double, const glm::vec4 &)
        """

    @overload
    def addVisualMesh(self, name: str, meshFile: str, scaleX: float = 1.0, scaleY: float = 1.0, scaleZ: float = 1.0, colorR: float = 1.0, colorG: float = 1.0, colorB: float = 1.0, colorA: float = 1.0) -> Visuals:
        """
        /**
           * @brief Add a visual mesh (not part of raisim::World).
           *
           * The file is imported through the renderer's asset manager. With async mesh loading enabled (the
           * default) the visual is returned immediately and its geometry appears once
           * pollAsyncMeshLoads() (called by every render) finalizes it.
           * @param name Visual name.
           * @param meshFile Mesh file path.
           * @param scaleX Mesh scale in X.
           * @param scaleY Mesh scale in Y.
           * @param scaleZ Mesh scale in Z.
           * @param colorR Red channel.
           * @param colorG Green channel.
           * @param colorB Blue channel.
           * @param colorA Alpha channel.
           * @return Shared pointer to the created visual.
           */
        C++: std::shared_ptr<Visuals> (const std::string &, const std::string &, double, double, double, float, float, float, float)
        """

    @overload
    def addVisualMesh(self, name: str, meshFile: str, scale: numpy.ndarray, color: numpy.ndarray = ...) -> Visuals:
        """
        /// @brief glm::vec3 scale + glm::vec4 color overload; prefer this in new code.
          /// @param name Visual name. @param meshFile Mesh file path. @param scale Per-axis scale.
          /// @param color RGBA color.
          /// @return Shared pointer to the created visual.
        C++: std::shared_ptr<Visuals> (const std::string &, const std::string &, const glm::dvec3 &, const glm::vec4 &)
        """

    @overload
    def importVisualScene(self, name: str, sceneFile: str, report: RayraiWindow.SceneImportReport | None = None, importLights: bool = True, lightIntensityScale: float = 1.0, scaleX: float = 1.0, scaleY: float = 1.0, scaleZ: float = 1.0, colorR: float = 1.0, colorG: float = 1.0, colorB: float = 1.0, colorA: float = 1.0) -> Visuals:
        """
        /**
           * @brief Import a scene file as a mesh visual and, optionally, its authored lights.
           *
           * Equivalent to addVisualMesh() plus importing the file's lights as
           * additional lights (dropping those beyond kMaxAdditionalLights). When lights
           * were imported and RenderQualitySettings::autoSelectImportedShadowLight is
           * set with a non-zero shadowedLightBudget, the dominant one is promoted to the
           * main shadow caster (promoteDominantAdditionalLightToMainShadowCaster()).
           * Light-import errors are reported in SceneImportReport::notes, not thrown.
           *
           * @param name Visual name (must be unique).
           * @param sceneFile Scene/mesh file path.
           * @param[out] report Optional import summary.
           * @param importLights Import the file's lights.
           * @param lightIntensityScale Multiplier applied to imported light intensities.
           * @param scaleX Mesh scale in X.
           * @param scaleY Mesh scale in Y.
           * @param scaleZ Mesh scale in Z.
           * @param colorR Red channel.
           * @param colorG Green channel.
           * @param colorB Blue channel.
           * @param colorA Alpha channel.
           * @return Shared pointer to the created visual.
           */
        C++: std::shared_ptr<Visuals> (const std::string &, const std::string &, SceneImportReport *, bool, float, double, double, double, float, float, float, float)
        """

    @overload
    def importVisualScene(self, name: str, sceneFile: str, opts: RayraiWindow.ImportVisualSceneOptions, report: RayraiWindow.SceneImportReport | None = None) -> Visuals: ...

    def makeVisualMesh(self, name: str, meshFile: str) -> RayraiWindow.VisualMeshBuilder:
        """
        /// @brief Start a fluent build of an `addVisualMesh` visual. See `VisualMeshBuilder`.
          /// @param name Visual name. @param meshFile Mesh file path.
          /// @return Builder bound to this renderer; call add() to create the visual.
        C++: VisualMeshBuilder (std::string, std::string)
        """

    @overload
    def addVisualPlane(self, name: str, width: float, height: float, colorR: float = 1.0, colorG: float = 1.0, colorB: float = 1.0, colorA: float = 1.0) -> Visuals:
        """
        /**
           * @brief Add a custom visual plane backed by the shared primitive plane mesh.
           *
           * Flat-shaded; the unit plane lies in the local XY plane with a +Z normal and
           * @p width / @p height scale it along local X / Y. Duplicate names are fatal.
           * @param name Visual name.
           * @param width Size along local X.
           * @param height Size along local Y.
           * @param colorR Red channel.
           * @param colorG Green channel.
           * @param colorB Blue channel.
           * @param colorA Alpha channel.
           * @return Shared pointer to the created visual.
           */
        C++: std::shared_ptr<Visuals> (const std::string &, double, double, float, float, float, float)
        """

    @overload
    def addVisualPlane(self, name: str, width: float, height: float, color: numpy.ndarray) -> Visuals:
        """
        /// @brief glm::vec4 color overload; prefer this in new code.
          /// @param name Visual name. @param width Size along local X. @param height Size along local Y.
          /// @param color RGBA color.
          /// @return Shared pointer to the created visual.
        C++: std::shared_ptr<Visuals> (const std::string &, double, double, const glm::vec4 &)
        """

    @overload
    def addVisualCustomMesh(self, name: str, meshes: Sequence[OpenGLMesh], color: numpy.ndarray = ...) -> Visuals:
        """
        /**
           * @brief Add a visual from custom OpenGL meshes.
           *
           * The mesh list is shared, not copied. Duplicate names are fatal.
           * @param name Visual name.
           * @param meshes Mesh list to render.
           * @param color Default color for non-textured meshes.
           * @return Shared pointer to the created visual.
           */
        C++: std::shared_ptr<Visuals> (const std::string &, const std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> &, const glm::vec4 &)
        """

    @overload
    def addVisualCustomMesh(self, name: str, meshes: Sequence[OpenGLMesh], color: numpy.ndarray, automaticMeshLodEnabled: bool) -> Visuals:
        """
        /**
           * @brief addVisualCustomMesh() variant that can disable automatic mesh LOD
           *        (for already prepared terrain LODs).
           *
           * Unlike the other overload this throws std::invalid_argument for a duplicate
           * or empty name, a null/empty mesh list, or a null mesh.
           * @param name Visual name.
           * @param meshes Mesh list to render (shared).
           * @param color Default color for non-textured meshes.
           * @param automaticMeshLodEnabled False to disable generic mesh simplification.
           * @return Shared pointer to the created visual.
           */
        C++: std::shared_ptr<Visuals> (const std::string &, const std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> &, const glm::vec4 &, bool)
        """

    def getVisualObject(self, name: str) -> Visuals:
        """
        /**
           * @brief Look up a visual by name.
           * @param name Visual name.
           * @return Shared pointer to the visual. Aborts (`RSFATAL`) if not found; use
           *         findVisualObject() for a non-fatal lookup.
           */
        C++: std::shared_ptr<Visuals> (const std::string &)
        """

    def findVisualObject(self, name: str) -> Visuals:
        """
        /// Look up a custom visual by name without aborting.
          /// @param name Visual name.
          /// @return The visual, or nullptr when it does not exist.
        C++: std::shared_ptr<Visuals> (const std::string &)
        """

    def removeVisualObject(self, name: str) -> None:
        """
        /**
           * @brief Remove a visual object by name.
           *
           * Clears the target visual if it was this one.
           * @param name Visual name to remove; aborts (`RSFATAL`) if unknown.
           */
        C++: void (const std::string &)
        """

    @overload
    def addInstancedVisuals(self, name: str, type: raisimpy.ShapeType, size: numpy.ndarray, color1: numpy.ndarray, color2: numpy.ndarray, meshFile: str = '') -> InstancedVisuals:
        """
        /**
           * @brief Create instanced visuals for repeated shapes or meshes.
           *
           * Instances are added through the returned object; each instance color is a
           * blend of @p color1 and @p color2 (see InstancedVisuals). Instanced names
           * form their own namespace and must be unique (`RSFATAL` otherwise).
           * Automatic mesh LOD is enabled.
           * @param name Visual name.
           * @param type Shape type.
           * @param size Base size for the instances.
           * @param color1 Primary instance color (linear RGBA).
           * @param color2 Secondary instance color (linear RGBA).
           * @param meshFile Mesh file for mesh instances.
           * @return Shared pointer to the created instanced visuals (also held by the renderer).
           */
        C++: std::shared_ptr<InstancedVisuals> (const std::string &, raisim::Shape::Type, const glm::vec3 &, const glm::vec4 &, const glm::vec4 &, const std::string &)
        """

    @overload
    def addInstancedVisuals(self, name: str, type: raisimpy.ShapeType, size: numpy.ndarray, color1: numpy.ndarray, color2: numpy.ndarray, meshFile: str, automaticMeshLodEnabled: bool) -> InstancedVisuals:
        """
        /// @copybrief addInstancedVisuals(const std::string&,raisim::Shape::Type,const glm::vec3&,const glm::vec4&,const glm::vec4&,const std::string&)
          /// @param name Visual name. @param type Shape type. @param size Base size.
          /// @param color1 Primary color. @param color2 Secondary color. @param meshFile Mesh file.
          /// @param automaticMeshLodEnabled False to disable automatic mesh simplification.
          /// @return Shared pointer to the created instanced visuals.
        C++: std::shared_ptr<InstancedVisuals> (const std::string &, raisim::Shape::Type, const glm::vec3 &, const glm::vec4 &, const glm::vec4 &, const std::string &, bool)
        """

    @overload
    def addInstancedVisuals(self, name: str, prototype: InstancedVisuals) -> InstancedVisuals:
        """
        /// Create an empty cell batch sharing a ready prototype's prepared meshes
          /// (see InstancedVisuals::cloneWithoutInstances()).
          /// @param name Batch name; throws std::invalid_argument if already used.
          /// @param prototype Fully prepared source batch.
          /// @return The new, empty batch.
        C++: std::shared_ptr<InstancedVisuals> (const std::string &, const InstancedVisuals &)
        """

    @overload
    def addInstancedVisuals(self, name: str, meshes: Sequence[OpenGLMesh], size: numpy.ndarray, color1: numpy.ndarray, color2: numpy.ndarray) -> InstancedVisuals:
        """
        /**
           * @brief Create instanced visuals from an already-created mesh list.
           * @param name Visual name (unique among instanced visuals).
           * @param meshes Mesh list shared by all instances.
           * @param size Base size for the instances.
           * @param color1 Primary instance color.
           * @param color2 Secondary instance color.
           * @return Shared pointer to the created instanced visuals.
           */
        C++: std::shared_ptr<InstancedVisuals> (const std::string &, const std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> &, const glm::vec3 &, const glm::vec4 &, const glm::vec4 &)
        """

    @overload
    def addInstancedVisuals(self, name: str, meshes: Sequence[OpenGLMesh], size: numpy.ndarray, color1: numpy.ndarray, color2: numpy.ndarray, automaticMeshLodEnabled: bool) -> InstancedVisuals:
        """
        /// Mesh-list overload with explicit automatic-LOD control.
          /// @param name Visual name. @param meshes Shared mesh list. @param size Base size.
          /// @param color1 Primary color. @param color2 Secondary color.
          /// @param automaticMeshLodEnabled False to disable automatic mesh simplification.
          /// @return Shared pointer to the created instanced visuals.
        C++: std::shared_ptr<InstancedVisuals> (const std::string &, const std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> &, const glm::vec3 &, const glm::vec4 &, const glm::vec4 &, bool)
        """

    def getInstancedVisuals(self, name: str) -> InstancedVisuals:
        """
        /**
           * @brief Look up instanced visuals by name.
           * @param name Visual name.
           * @return Shared pointer to the instanced visuals. Aborts if not found.
           */
        C++: std::shared_ptr<InstancedVisuals> (const std::string &)
        """

    def findInstancedVisuals(self, name: str) -> InstancedVisuals:
        """
        /**
           * @brief Look up instanced visuals by name without aborting when absent.
           * @param name Visual name.
           * @return The instanced visuals, or nullptr.
           */
        C++: std::shared_ptr<InstancedVisuals> (const std::string &)
        """

    def getTendonVisual(self, name: str) -> InstancedVisuals:
        """
        /** @brief Automatically maintained spatial tendon drawing, available after scene sync.
           * Returns nullptr for an unknown or fixed tendon. Width is a radius; color,
           * enabled state and path changes are reflected on the next rendered frame.
           * @param name Tendon name.
           * @return Renderer-maintained instanced visual, or nullptr. */
        C++: std::shared_ptr<InstancedVisuals> (const std::string &) const
        """

    def removeInstancedVisuals(self, name: str) -> None:
        """
        /**
           * @brief Remove instanced visuals by name.
           * @param name Visual name to remove; aborts (`RSFATAL`) if unknown.
           */
        C++: void (const std::string &)
        """

    def addCameraFrustum(self, name: str, color: numpy.ndarray = ...) -> CameraFrustum:
        """
        /**
           * @brief Create a camera frustum visualization.
           *
           * Adds a line-mesh custom visual named @p name (remove it with
           * removeVisualObject()); duplicate names are fatal.
           * @param name Visual name.
           * @param color Frustum color.
           * @return Shared pointer to the frustum helper used to update its shape and pose.
           */
        C++: std::shared_ptr<CameraFrustum> (const std::string &, const glm::vec4 &)
        """

    def setGroundPatternResourcePath(self, path: str) -> None:
        """
        /**
           * @brief Set a tiled ground texture for halfspace visuals.
           *
           * Loaded with loadTextureWithTiling() (linear, not sRGB). GL context required.
           * @param path Path to the texture image.
           */
        C++: void (const std::string &)
        """

    def getGroundPatternResourcePath(self) -> str:
        """
        /**
           * @brief Get the current ground texture path.
           * @return Ground texture path string.
           */
        C++: std::string ()
        """

    def updateObjectLists(self) -> None:
        """
        /**
           * @brief Copy the bound World's render state for the next renders.
           *
           * Captures the World's objects, poses, appearance, terrain and deformable
           * geometry, tendons and .rscene scene into a PhysicsSnapshot and applies it.
           * Every render calls this automatically; renders draw the copy, not the World
           * (see ThreadingMode for stepping the World on another thread). Does nothing
           * while an applyPhysicsSnapshot() snapshot is shown. Requires a current GL context.
           */
        C++: void ()
        """

    def applyPhysicsSnapshot(self, snapshot: PhysicsSnapshot) -> None:
        """
        /** Show a physics snapshot instead of the bound World.
           *
           * Capture with PhysicsSnapshotCapture immediately after integrate2() while the World is
           * protected, publish the immutable snapshot to the render thread without holding a render
           * mutex on the physics thread, and apply it on the render thread with its GL context
           * current. Later renders draw it, including shadow maps, until the next applied snapshot
           * or clearPhysicsSnapshot(); they never read a World. The snapshot can outlive removed
           * physics objects. Scene files and materials must remain available.
           * @param snapshot Frame to show; must be non-null.
           */
        C++: void (std::shared_ptr<const PhysicsSnapshot>)
        """

    def clearPhysicsSnapshot(self) -> None:
        """
        /// Show the bound World again: copies it now, as updateObjectLists() does, so protect it as
          /// for a render. Requires a current GL context.
        C++: void ()
        """

    def setExplosionEffectSettings(self, settings: ExplosionEffectSettings) -> None:
        """
        /** @name Explosions
           *  The window draws the explosions, projectiles and projectile hits of the world it shows
           *  (World::addExplosion(), World::addProjectile()): a fireball and smoke for each explosion, a
           *  muzzle flash where a projectile was launched, a tracer behind it in flight, and sparks where an
           *  inert projectile hits. addExplosionEffect() draws an explosion that is not in the world and
           *  addFireEffect() a fire. See ExplosionEffectSettings for what is drawn and the clock effects follow. */
          ///@{
          /// Choose how explosions are drawn (values are clamped to their documented ranges).
        C++: void (const ExplosionEffectSettings &)
        """

    def getExplosionEffectSettings(self) -> ExplosionEffectSettings:
        """
        /// @return How explosions are drawn.
        C++: const ExplosionEffectSettings &() const
        """

    def addExplosionEffect(self, effect: ExplosionEffect) -> None:
        """
        /// Draw an explosion starting now (by the clock of ExplosionEffectSettings::useWorldTime).
          /// @param effect Where it is and how large.
        C++: void (const ExplosionEffect &)
        """

    def addFireEffect(self, effect: FireEffect) -> None:
        """
        /// Start a fire burning now (by the clock of ExplosionEffectSettings::useWorldTime).
          /// @param effect Where it burns, how large and for how long.
        C++: void (const FireEffect &)
        """

    def clearExplosionEffects(self) -> None:
        """
        /// Stop drawing every explosion, fire, muzzle flash and impact now in progress.
        C++: void ()
        """

    def activeExplosionEffectCount(self) -> int:
        """
        /// @return Number of explosions, fires, muzzle flashes and impacts in progress, scorch marks included.
        C++: size_t () const
        """

    def prepareExplosionEffects(self) -> None:
        """
        /// Compile the shaders and create the textures explosions draw with now, so that the first explosion
          /// does not stall a frame (most noticeably in the browser): the effects' own shaders and, with scorch
          /// marks, the post-processing variants that projected decals select for the main camera, which the
          /// next post-processed renders also draw once each with hidden decals, so that the browser builds their
          /// pipelines. From then on, foliage drawn with a sun-only program also links the program an
          /// explosion's light selects, when it is first drawn. Requires a current GL context.
        C++: void ()
        """

    def setObjectHidden(self, object: raisimpy.Object | None, hidden: bool) -> None:
        """
        ///@}
          /// Change render visibility without removing an object's physics collision.
          /// @param object Object in this renderer's world; throws std::invalid_argument
          ///        when null or not part of the world.
          /// @param hidden True to stop drawing it.
        C++: void (raisim::Object *, bool)
        """

    def debugSceneObjectCandidateRevision(self) -> int:
        """
        /**
           * @brief Return the object candidate cache revision for tests and diagnostics.
           * @return Counter value.
           */
        C++: uint64_t () const
        """

    def debugSceneVisualizationCandidateRebuilds(self) -> int:
        """
        /**
           * @brief Return visualization candidate cache rebuild count for tests and diagnostics.
           * @return Counter value.
           */
        C++: size_t () const
        """

    def debugSceneVisualizationCandidateCacheHits(self) -> int:
        """
        /**
           * @brief Return visualization candidate cache hit count for tests and diagnostics.
           * @return Counter value.
           */
        C++: size_t () const
        """

    def debugActiveAdditionalLightSelectionRevision(self) -> int:
        """
        /**
           * @brief Return active additional-light selection revision for tests and diagnostics.
           * @return Counter value.
           */
        C++: uint64_t () const
        """

    def debugActiveAdditionalLightSelectionRebuilds(self) -> int:
        """
        /**
           * @brief Return active additional-light selection rebuild count for tests and diagnostics.
           * @return Counter value.
           */
        C++: size_t () const
        """

    def debugActiveAdditionalLightSelectionCacheHits(self) -> int:
        """
        /**
           * @brief Return active additional-light selection cache hit count for tests and diagnostics.
           * @return Counter value.
           */
        C++: size_t () const
        """

    def orthogonalViewSwitch(self) -> None:
        """
        /**
           * @brief Step the viewer camera through axis-aligned orthographic views.
           *
           * Switches the internal camera to orthographic projection and, on each call,
           * moves it to the next of six views placed 2 m from the camera target on the
           * +X, +Y, -X, -Y, +Z and -Z sides, looking at the target. It does not switch
           * back to perspective; use getCamera().setProjectionMode() for that.
           */
        C++: void ()
        """

    @overload
    def renderDepthPlaneDistance(self, sensorCam: raisimpy.DepthCamera, cam: Camera, depthPlaneOverride: Shader | None = None, drawVisualizationObjects: bool = False, visualizationObjectsMustBeDetectable: bool = True) -> None:
        """
        /**
           * @brief Render a linear depth plane for a raisim depth camera.
           *
           * Copies the sensor's pose into @p cam (as renderWithExternalCamera() does),
           * then behaves like the Camera overload.
           * @param sensorCam Raisim depth camera; its pose is updated and copied.
           * @param cam Render camera to use; receives the depth image.
           * @param depthPlaneOverride Optional shader override.
           * @param drawVisualizationObjects Include custom and instanced visuals.
           * @param visualizationObjectsMustBeDetectable When true, include only visualization objects
           *        marked detectable; when false, include all visualization objects.
           */
        C++: void (raisim::DepthCamera &, Camera &, Shader *, bool, bool)
        """

    @overload
    def renderDepthPlaneDistance(self, cam: Camera, depthPlaneOverride: Shader | None = None, drawVisualizationObjects: bool = False, visualizationObjectsMustBeDetectable: bool = True) -> None:
        """
        /**
           * @brief Render a linear depth plane with a given camera.
           *
           * Writes camera-plane depth into @p cam's R32F linear-depth target (read it
           * with Camera::getLinearDepthTexture()); pixels with no geometry are 0.
           * Weather, post-processing and point clouds are not involved. Requires this
           * renderer's GL context.
           * @param cam Render camera to use; receives the depth image.
           * @param depthPlaneOverride Optional shader override.
           * @param drawVisualizationObjects Include custom and instanced visuals.
           * @param visualizationObjectsMustBeDetectable When true, include only visualization objects
           *        marked detectable; when false, include all visualization objects.
           */
        C++: void (Camera &, Shader *, bool, bool)
        """

    @overload
    def renderSegmentation(self, sensorCam: raisimpy.SegmentationCamera, cam: Camera, drawVisualizationObjects: bool = False, visualizationObjectsMustBeDetectable: bool = True) -> None:
        """
        /**
           * @brief Render a segmentation image for a raisim segmentation camera.
           *
           * Copies the sensor's pose into @p cam (as renderWithExternalCamera() does), then behaves like
           * the Camera overload. Read the images with Camera::getRawImage(raisim::SegmentationCamera&).
           * @param sensorCam Raisim segmentation camera; its pose is updated and copied.
           * @param cam Render camera to use; receives the segmentation image.
           * @param drawVisualizationObjects Include custom and instanced visuals.
           * @param visualizationObjectsMustBeDetectable When true, include only visualization objects
           *        marked detectable; when false, include all visualization objects.
           */
        C++: void (raisim::SegmentationCamera &, Camera &, bool, bool)
        """

    @overload
    def renderSegmentation(self, cam: Camera, drawVisualizationObjects: bool = False, visualizationObjectsMustBeDetectable: bool = True) -> None:
        """
        /**
           * @brief Render a segmentation image with a given camera.
           *
           * Writes (world index, body index, label, 0) per pixel into @p cam's RGBA32I segmentation target
           * (Camera::getSegmentationTexture(), Camera::readSegmentation()); pixels with no geometry hold
           * raisim::SegmentationCamera::kBackground (-1) in every channel. The pass draws the geometry of
           * renderDepthPlaneDistance(), so segmentation and depth images of a camera agree pixel for pixel,
           * without antialiasing, blending, weather or post-processing. Visuals write their
           * Visuals::getSegmentationIds(). A fisheye lens gets the color image's remap, sampled at the
           * nearest pixel. Requires this renderer's GL context.
           * @param cam Render camera to use; receives the segmentation image.
           * @param drawVisualizationObjects Include custom and instanced visuals.
           * @param visualizationObjectsMustBeDetectable When true, include only visualization objects
           *        marked detectable; when false, include all visualization objects.
           */
        C++: void (Camera &, bool, bool)
        """

    def getDepthPlaneTexture(self) -> int:
        """
        /**
           * @brief Get the internal viewer camera's linear depth texture id.
           *
           * For an external camera passed to renderDepthPlaneDistance(), read that
           * camera's Camera::getLinearDepthTexture() instead.
           * @return OpenGL texture id for R32F linear depth (0 before it is allocated).
           */
        C++: unsigned int () const
        """

    def measureSpinningLidarSingleDrawGPU(self, lidar: raisimpy.SpinningLidar, sensorPosW: numpy.ndarray, sensorRotW: numpy.ndarray, objectToExclude: raisimpy.Object | None = None) -> None:
        """
        /**
           * @brief Measure a spinning LiDAR on the GPU.
           *
           * Advances the head with SpinningLidar::advanceSweep() (the sweep state is shared with
           * SpinningLidar::update(), so the two can alternate) and casts the swept yaw columns
           * against the world's collision bodies
           * (World::getRayCastBodies()), stores the hits in the sensor frame with
           * lidar.setScan() and stamps the world time. The scan reproduces
           * SpinningLidar::update(): the same columns and point order, exact intersections
           * with primitives, heightmaps and meshes, rays starting at rangeMin, and the
           * sensor's own body ignored. Ranges agree to float precision (about 1e-5 of the
           * range); a ray grazing a silhouette within that tolerance may differ.
           * Requires this renderer's GL context.
           *
           * @param lidar Raisim spinning lidar to update.
           * @param sensorPosW LiDAR position in world coordinates.
           * @param sensorRotW LiDAR orientation in world coordinates.
           * @param objectToExclude Optional object whose bodies are all ignored (e.g. the carrier).
           * @throws std::runtime_error if a GL resource cannot be created or the scene exceeds a GPU limit.
           */
        C++: void (raisim::SpinningLidar &, const glm::dvec3 &, const glm::dmat3 &, const raisim::Object *)
        """

    def setLinearHdrRenderingEnabled(self, enabled: bool) -> None:
        """
        /** @brief Keep scene radiance linear through MSAA, fog, bloom and DoF, then apply
           * the selected exposure/tone curve/gamma in the existing post pass.
           * Opt-in for compatibility. External captures with postProcess=false keep
           * their legacy contract; internal reflection captures inherit the scene domain.
           * Bloom thresholds are scene-linear radiance in this mode. Changing it resets
           * temporal AA history.
           * @param enabled True to enable linear HDR rendering (default false). */
        C++: void (bool)
        """

    def linearHdrRenderingEnabled(self) -> bool:
        """
        /// @return True when linear HDR rendering is enabled (see setLinearHdrRenderingEnabled()).
        C++: bool () const
        """

    def geometryRefractionDiagnostics(self) -> GeometryRefractionDiagnostics:
        """
        /// @return Diagnostics: state of geometry (dielectric) refraction rendering; default-constructed
          /// when that path has never run. CPU only.
        C++: GeometryRefractionDiagnostics () const
        """

    def geometryRefractionSamplingStatistics(self) -> GeometryRefractionSamplingStatistics:
        """
        /// Explicit GPU readback; requires this renderer's GL context and may stall.
          /// @return Sampling statistics, default-constructed when the path has never run.
        C++: GeometryRefractionSamplingStatistics () const
        """

    def setSkyVisibilityGrid(self, grid: SkyVisibilityGrid, strength: float = 1.0, edgeFadeMeters: float = 2.0) -> None:
        """
        /**
           * @brief Upload a static sky-visibility grid that attenuates environment lighting.
           *
           * Call on the rendering thread with its GL context current; the data is
           * copied into a texture. Only global environment lighting is attenuated;
           * direct light, authored ambient, lightmaps and local irradiance volumes
           * keep their own visibility. Replaces any previous grid.
           * @param grid Visibility grid; validated (throws std::invalid_argument).
           * @param strength Attenuation strength in [0, 1] (throws std::invalid_argument otherwise).
           * @param edgeFadeMeters Fade distance at the grid boundary in metres, >= 0.
           */
        C++: void (const SkyVisibilityGrid &, float, float)
        """

    def clearSkyVisibilityGrid(self) -> None:
        """
        /// Remove the sky-visibility grid.
        C++: void ()
        """

    def setBakedIrradianceGrid(self, grid: BakedIrradianceGrid, strength: float = 1.0, edgeFadeMeters: float = 2.0) -> None:
        """
        /**
           * @brief Upload a static baked-irradiance grid (scene-referred diffuse bounce).
           *
           * Independent of environment sky fill. Upload/clear on the owning rendering
           * GL context; disabled by default. Replaces any previous grid and resets
           * temporal AA history.
           * @param grid Irradiance grid; validated (throws std::invalid_argument).
           * @param strength Strength in [0, 4] (throws std::invalid_argument otherwise).
           * @param edgeFadeMeters Fade distance at the grid boundary in metres, >= 0.
           */
        C++: void (const BakedIrradianceGrid &, float, float)
        """

    def clearBakedIrradianceGrid(self) -> None:
        """
        /// Remove the baked-irradiance grid.
        C++: void ()
        """

    def getLight(self) -> Light:
        """
        /**
           * @brief Access the main scene light (the single main shadow caster).
           *
           * Direct edits take effect on the next frame, but setRenderQualitySettings(),
           * setRenderQualityPreset(), promoteDominantAdditionalLightToMainShadowCaster()
           * and enabled weather overwrite parts of it. Call requestShadowUpdate() after
           * moving it when shadows are not refreshed every frame.
           * @return Reference to the main light, owned by the renderer.
           */
        C++: Light &()
        """

    def addAdditionalLight(self, light: AdditionalLight) -> None:
        """
        /**
           * @brief Add a light besides the main light.
           *
           * The light is sanitized and copied. Additional lights with
           * AdditionalLight::castsShadows set can receive shadow maps when shadows are
           * enabled and RenderQualitySettings::shadowedLightBudget is greater than 1
           * (up to kMaxAdditionalShadowLights per frame); otherwise they only shade.
           * Note that setRenderQualitySettings() / setRenderQualityPreset() clear all
           * additional lights, so add lights after choosing the quality settings.
           * @param light Additional light. Ignored once kMaxAdditionalLights lights exist.
           */
        C++: void (const AdditionalLight &)
        """

    def updateAdditionalLight(self, index: int, light: AdditionalLight) -> bool:
        """
        /**
           * @brief Replace an existing additional light without forcing unrelated shadow maps dirty.
           * @param index Index in insertion order.
           * @param light New light (sanitized and copied).
           * @return True if the light existed and was replaced.
           */
        C++: bool (size_t, const AdditionalLight &)
        """

    def clearAdditionalLights(self) -> None:
        """
        /**
           * @brief Remove all additional lights (including their shadow maps).
           *
           * The main light is unaffected.
           */
        C++: void ()
        """

    def additionalLightCount(self) -> int:
        """
        /**
           * @brief Return the current number of additional lights.
           * @return Number of additional lights (excluding the main light).
           */
        C++: size_t () const
        """

    def promoteDominantAdditionalLightToMainShadowCaster(self, preferSpotLights: bool = True, removePromotedAdditionalLight: bool = True) -> bool:
        """
        /**
           * @brief Promote the dominant imported/local light to the single shadow-casting main light.
           *
           * This is intended for authored glTF scenes that have a clear lamp/spot light. It keeps
           * low-quality rendering fast because no additional shadow maps are created. Candidates must
           * have AdditionalLight::castsShadows set and non-zero power; point lights are never promoted
           * (they use cubemap shadows through the additional-light budget). The main light takes over
           * the candidate's type, pose, color and attenuation, and its shadows are enabled.
           *
           * @param preferSpotLights Prefer spot lights over other light types when possible.
           * @param removePromotedAdditionalLight Remove the promoted light from the additional-light list
           *        to avoid double-counting its direct contribution.
           * @return True if a suitable light was promoted.
           */
        C++: bool (bool, bool)
        """

    def importSceneLights(self, sceneFile: str, intensityScale: float = 1.0) -> int:
        """
        /**
           * @brief Import authored scene lights from a mesh/scene file through Assimp.
           *
           * Imported lights are appended as additional lights (see addAdditionalLight()).
           * This is explicit so low-quality rendering does not pay for scene lights unless
           * an application imports them. When any light was added and
           * RenderQualitySettings::autoSelectImportedShadowLight is set with a non-zero
           * shadowedLightBudget, the dominant one is promoted to the main shadow caster.
           * Import errors are printed to stderr, not thrown.
           *
           * @param sceneFile Scene/mesh file to read lights from.
           * @param intensityScale Multiplier applied to imported light intensities.
           * @return Number of lights appended, capped by kMaxAdditionalLights.
           */
        C++: size_t (const std::string &, float)
        """

    def setRenderQualityPreset(self, preset: RenderQualityPreset) -> None:
        """
        /**
           * @brief Apply a named viewer-quality preset at runtime.
           *
           * Equivalent to setRenderQualitySettings(defaultRenderQualitySettings(preset))
           * followed by recording @p preset as the active preset; the same side effects apply.
           * @param preset Quality preset to apply.
           */
        C++: void (RenderQualityPreset)
        """

    def getRenderQualityPreset(self) -> RenderQualityPreset:
        """
        /**
           * @brief Return the active named quality preset, or Custom after explicit settings.
           * @return Preset; Fast for a freshly constructed renderer.
           */
        C++: RenderQualityPreset () const
        """

    def setRenderQualitySettings(self, settings: RenderQualitySettings) -> None:
        """
        /**
           * @brief Apply explicit viewer-quality settings at runtime.
           *
           * Besides storing (and clamping) the settings, this re-derives renderer state
           * from them: background color, fog density, gamma, depth of field, the main
           * light (type reset to directional, direction, colors, shadow resolution and
           * parameters), the shadow box and center offset, and texture filtering of
           * loaded textures. It also **clears all additional lights** and, when
           * RenderQualitySettings::addViewerFillLights is set, adds the built-in fill
           * and rim lights. The active preset becomes Custom. With weather enabled the
           * settings become the base that weather is applied on.
           * @param settings Settings to apply. Throws std::invalid_argument for an
           *        unknown temporal AA method, a non-finite jitter scale, a current
           *        weight outside [0.02, 1] or a sharpness outside [0, 1].
           */
        C++: void (const RenderQualitySettings &)
        """

    def getRenderQualitySettings(self) -> RenderQualitySettings:
        """
        /**
           * @brief Return the active explicit viewer-quality settings.
           * @return Effective settings, including weather overrides and clamping.
           */
        C++: const RenderQualitySettings &() const
        """

    def getUserRenderQualitySettings(self) -> RenderQualitySettings:
        """
        /// @return Settings as last set by the user, i.e. without weather overrides while
          /// weather is enabled; otherwise the same as getRenderQualitySettings().
        C++: const RenderQualitySettings &() const
        """

    def setDepthPrepassesEnabled(self, opaque: bool, foliageAlpha: bool) -> None:
        """
        /** @brief Toggle depth prepasses while preserving authored lights and materials.
           *
           * Unlike setRenderQualitySettings() this does not reset lights; the preset becomes Custom.
           * @param opaque RenderQualitySettings::opaqueDepthPrepass.
           * @param foliageAlpha RenderQualitySettings::foliageAlphaDepthPrepass. */
        C++: void (bool, bool)
        """

    def setAdditionalLightContributionThreshold(self, threshold: float) -> None:
        """
        /// Set the total additional-light incident-radiance cutoff budget without
          /// resetting authored lights. Zero (default) keeps exact unlimited lighting.
          /// Invalid/negative values disable culling. This is not a final-pixel tolerance.
          /// The preset becomes Custom.
          /// @param threshold RenderQualitySettings::additionalLightContributionThreshold.
        C++: void (float)
        """

    def setFoliageWindTimeSeconds(self, seconds: float) -> None:
        """
        /**
           * @brief Set the global foliage wind animation clock without changing quality preset state.
           * @param seconds Clock value in seconds; negative or non-finite values become 0.
           */
        C++: void (float)
        """

    def advanceFoliageWindTime(self, deltaSeconds: float) -> None:
        """
        /**
           * @brief Advance the global foliage wind animation clock.
           * @param deltaSeconds Elapsed seconds; non-positive or non-finite values are ignored.
           *        The clock wraps at 100000 s.
           */
        C++: void (float)
        """

    def printRenderDiagnostics(self) -> tuple:
        """
        /**
           * @brief Print active viewer-quality, light, shadow, and post-process diagnostics.
           * @param out Output stream.
           */
        C++: void (std::ostream &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def writeRenderDiagnosticsJson(self, importReport: RayraiWindow.SceneImportReport | None = None) -> tuple:
        """
        /**
           * @brief Write structured render diagnostics as JSON.
           *
           * The JSON is intended for regression tests, issue reports, and imported-scene debugging.
           * It contains render-quality settings, scene/resource counts, lights, shadow selection state,
           * postprocess state, and an optional scene-import report.
           * @param out Output stream.
           * @param importReport Optional report to embed (not stored).
           */
        C++: void (std::ostream &, const SceneImportReport *) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def renderDiagnosticsJson(self, importReport: RayraiWindow.SceneImportReport | None = None) -> str:
        """
        /**
           * @brief Return structured render diagnostics as a JSON string.
           * @param importReport Optional report to embed.
           * @return Same JSON as writeRenderDiagnosticsJson().
           */
        C++: std::string (const SceneImportReport *) const
        """

    def estimateRenderPassAccounting(self) -> RayraiWindow.RenderPassAccounting:
        """
        /**
           * @brief Return estimated render-pass accounting for the current quality settings.
           *
           * This is an estimate-only diagnostic. It does not insert GPU timers or render any frame.
           * Use captureRenderPassTimings() when exact blocking GPU timings are required.
           * @return Estimated pass counts.
           */
        C++: RenderPassAccounting () const
        """

    @overload
    def transparentDrawDebugView(self, maxItems: int = 32) -> RayraiWindow.TransparentDrawDebugView:
        """
        /**
           * @brief Return the transparent draw order that the viewer path should use for the current camera.
           *
           * The result is diagnostic-only. It does not render and does not mutate instance buffers.
           * @param maxItems Maximum entries in backToFrontOrder; 0 returns all.
           * @return Transparent draw diagnostics for the internal camera.
           */
        C++: TransparentDrawDebugView (size_t) const
        """

    @overload
    def transparentDrawDebugView(self, camera: Camera, maxItems: int = 32) -> RayraiWindow.TransparentDrawDebugView:
        """
        /**
           * @brief Return transparent draw diagnostics for an explicit render camera.
           * @param camera Camera used for sorting and the overdraw estimate.
           * @param maxItems Maximum entries in backToFrontOrder; 0 returns all.
           * @return Transparent draw diagnostics.
           */
        C++: TransparentDrawDebugView (const Camera &, size_t) const
        """

    def shaderWarmupDiagnostics(self) -> RayraiWindow.ShaderWarmupDiagnostics:
        """
        /**
           * @brief Return startup shader compile/warmup diagnostics for built-in RayRai programs.
           *
           * RayRai registers all built-in shader programs during construction and compiles
           * optional shaders lazily on first use. This exposes the measured compile/link cost
           * and linked program count so startup and first-use stalls can be tracked.
           * @return Warmup diagnostics (queries GL program state; GL context required).
           */
        C++: ShaderWarmupDiagnostics () const
        """

    def linkedShaderNames(self) -> list[str]:
        """
        /**
           * @brief Names of every shader program currently linked, sorted alphabetically.
           * Useful for diagnostics and for callers that want to record the "hot set" of
           * shaders an application actually uses so a future run can pre-warm only those.
           * @return Sorted program names.
           */
        C++: std::vector<std::string> () const
        """

    def warmupNextShader(self) -> tuple:
        """
        /**
           * @brief Compile one not-yet-compiled built-in shader program, if any.
           *
           * Returns the number of milliseconds spent compiling (0 if no work was done).
           * The output flag `done` is set to true when every registered program has been
           * compiled at least once. Iterates all registered programs — most of which an
           * application typically doesn't use; prefer `compileShaderByName` to pre-warm
           * just the heavy shaders your active scene actually exercises.
           * Must be called on the thread that owns the GL context.
           * @param[out] done Set to true when nothing was left to compile, false otherwise.
           * @return Milliseconds spent compiling. Throws std::runtime_error when a shader fails.
           */
        C++: long long (bool &)
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def compileShaderByName(self, name: str) -> int:
        """
        /**
           * @brief Compile a specific built-in shader by name if not already linked.
           * Returns the milliseconds spent compiling (0 if already linked or unknown name).
           * Lets the application warm just the shaders it knows will be needed —
           * e.g. `pbrMeshHigh` (the high-fidelity PBR mesh program) is typically the dominant
           * cost for content rendering, so pre-warming only that one avoids most of the visible
           * "freeze" on the first content frame without paying for unused shader variants.
           * Must be called on the thread that owns the GL context.
           * @param name Built-in program name (see linkedShaderNames()).
           * @return Milliseconds spent compiling. Throws std::runtime_error when the shader fails.
           */
        C++: long long (const std::string &)
        """

    def recommendMaterialTextureBudget(self, budgetBytes: int = 0) -> RayraiWindow.MaterialTextureBudgetRecommendation:
        """
        /**
           * @brief Build a non-mutating material texture memory fallback recommendation.
           *
           * If budgetBytes is zero, a conservative preset-aware default is used. The recommendation
           * only changes quality settings that reduce texture sampling/setup cost; it does not unload
           * existing textures or alter sensor behavior. Texture sizes are queried from GL, so call it
           * with this renderer's GL context current.
           * @param budgetBytes Texture memory budget in bytes; 0 selects the preset default.
           * @return Recommendation; apply recommendedSettings yourself if desired.
           */
        C++: MaterialTextureBudgetRecommendation (size_t) const
        """

    def writeRenderDiagnosticsFiles(self, directory: str | os.PathLike, importReport: RayraiWindow.SceneImportReport | None = None) -> None:
        """
        /**
           * @brief Write structured diagnostics files, including uncapped mesh/material tables.
           *
           * This is intended for explicit issue reports and offline analysis. The regular
           * in-memory JSON remains capped so Fast/Balanced users do not accidentally pay for
           * large debug payloads unless they request file export.
           * @param directory Output directory, created if needed; files such as
           *        `render_diagnostics.json` and `mesh_diagnostics_full.json` are overwritten.
           * @param importReport Optional report to embed.
           */
        C++: void (const std::filesystem::path &, const SceneImportReport *) const
        """

    @staticmethod
    def defaultRenderQualitySettings(preset: RenderQualityPreset) -> RenderQualitySettings:
        """
        /**
           * @brief Return the default settings for a named preset.
           * @param preset Preset to query.
           * @return Settings that setRenderQualityPreset() would apply.
           */
        C++: RenderQualitySettings (RenderQualityPreset)
        """

    @staticmethod
    def recommendDynamicQuality(settings: RenderQualitySettings, preset: RenderQualityPreset, timings: Sequence[RayraiWindow.RenderPassTiming]) -> RayraiWindow.DynamicQualityRecommendation:
        """
        /**
           * @brief Build a sensor-safe dynamic-quality recommendation from explicit GPU pass timings.
           *
           * This does not mutate renderer state. It is intended for Fast/Balanced viewer loops that
           * explicitly opt into adaptive render scale or static shadow updates after measuring GPU cost.
           * High/Ultra recommendations never change settings. Otherwise, when the summed GPU time exceeds
           * the preset budget, shadow-bound frames stop per-frame shadow updates, fill-bound frames lower
           * viewerRenderScale by 15 % (not below 0.6 for Fast, 0.75 otherwise), and post-process-bound
           * frames reduce bloom/AO/indirect-lighting cost.
           * @param settings Current settings.
           * @param preset Preset that selects the budget (see DynamicQualityRecommendation::targetFrameMs).
           * @param timings Pass timings, e.g. from captureRenderPassTimings().
           * @return Recommendation.
           */
        C++: DynamicQualityRecommendation (const RenderQualitySettings &, RenderQualityPreset, const std::vector<RenderPassTiming> &)
        """

    @staticmethod
    def analyzeRgbaLuminance(rgba: Sequence[int], width: int, height: int) -> RayraiWindow.RgbaLuminanceMetrics:
        """
        /**
           * @brief Analyze display-space luminance and saturation from an RGBA8 capture.
           *
           * This is intended for exposure/tone-map diagnostics and image regression tests.
           * It does not render or read from OpenGL; callers pass already captured RGBA8 pixels.
           * @param rgba RGBA8 pixels (alpha ignored); at most width * height pixels are read.
           * @param width Image width in pixels.
           * @param height Image height in pixels.
           * @return Metrics; zero-filled for empty or invalid input.
           */
        C++: RgbaLuminanceMetrics (const std::vector<unsigned char> &, int, int)
        """

    @staticmethod
    def recommendExposure(metrics: RayraiWindow.RgbaLuminanceMetrics, currentExposure: float = 1.0, targetMedianLuminance: float = 0.45, targetP95Luminance: float = 0.9, minExposure: float = 0.125, maxExposure: float = 8.0) -> RayraiWindow.ExposureRecommendation:
        """
        /**
           * @brief Build a bounded exposure recommendation from display-space luminance metrics.
           *
           * This does not mutate renderer state. It is intended for explicit viewer calibration,
           * quality-comparison reports, and offline screenshot tooling. Sensor renders should only
           * apply the result if the caller explicitly opts into viewer-style postprocessing.
           *
           * Highlights take priority: clipping (> 1 % channels) or p95 above target scales the
           * exposure down to the p95 target; otherwise midtones below 70 % / above 135 % of the
           * median target are corrected toward it without pushing p95 past its target.
           * @param metrics Metrics of an image rendered at @p currentExposure.
           * @param currentExposure Exposure used for that image; must be positive.
           * @param targetMedianLuminance Desired median display luminance.
           * @param targetP95Luminance Desired 95th percentile display luminance.
           * @param minExposure Lower clamp for the result; must be positive.
           * @param maxExposure Upper clamp for the result.
           * @return Recommendation; `valid` is false for invalid input.
           */
        C++: ExposureRecommendation (const RgbaLuminanceMetrics &, double, double, double, double, double)
        """

    @staticmethod
    def smoothExposure(currentExposure: float, targetExposure: float, deltaSeconds: float, brightenRate: float = 1.2, darkenRate: float = 3.0, minExposure: float = 0.125, maxExposure: float = 8.0) -> float:
        """
        /**
           * @brief Smooth an explicitly chosen exposure target over time.
           *
           * This helper is stateless and does not mutate renderer settings. Use a larger darken
           * rate than brighten rate to adapt quickly away from clipping while avoiding flicker
           * when lifting dark scenes. Computes `current + (target - current) * (1 - exp(-rate * dt))`
           * with both exposures clamped to [minExposure, maxExposure].
           * @param currentExposure Current exposure.
           * @param targetExposure Desired exposure.
           * @param deltaSeconds Elapsed time in seconds.
           * @param brightenRate Rate (1/s) used when the target is brighter.
           * @param darkenRate Rate (1/s) used when the target is darker.
           * @param minExposure Lower exposure bound.
           * @param maxExposure Upper exposure bound.
           * @return New exposure; the clamped current exposure when any input is invalid.
           */
        C++: double (double, double, double, double, double, double, double)
        """

    @staticmethod
    def analyzeRgbaLuminanceHistogram(rgba: Sequence[int], width: int, height: int, binCount: int = 64) -> RayraiWindow.RgbaLuminanceHistogram:
        """
        /**
           * @brief Build a display-space luminance histogram from an RGBA8 capture.
           *
           * This is a CPU diagnostics helper for exposure reports and regression artifacts.
           * It does not render, read OpenGL state, or mutate color-management settings.
           * @param rgba RGBA8 pixels.
           * @param width Image width in pixels.
           * @param height Image height in pixels.
           * @param binCount Number of equal-width bins over luminance [0, 1].
           * @return Histogram; empty for invalid input.
           */
        C++: RgbaLuminanceHistogram (const std::vector<unsigned char> &, int, int, size_t)
        """

    @staticmethod
    def renderLuminanceHistogramRgba(histogram: RayraiWindow.RgbaLuminanceHistogram, width: int, height: int) -> list[int]:
        """
        /**
           * @brief Render a luminance histogram into a simple RGBA8 debug image.
           *
           * The output is top-left row order. It is intended for reports and screenshots,
           * not for the main frame path.
           * @param histogram Histogram to draw.
           * @param width Output width in pixels.
           * @param height Output height in pixels.
           * @return RGBA8 image of width * height pixels, or empty for a non-positive size.
           */
        C++: std::vector<unsigned char> (const RgbaLuminanceHistogram &, int, int)
        """

    @staticmethod
    def renderCloudDebugRgba(settings: RenderQualitySettings, width: int, height: int) -> list[int]:
        """
        /**
           * @brief Render procedural cloud diagnostics into a three-panel RGBA8 image.
           *
           * The panels are coverage mask, projected shadow attenuation, and approximate cloud-lighting
           * contribution. The CPU path mirrors the shader-side procedural cloud mask, so tests and
           * showcase images can inspect the weather cloud path without screenshots alone.
           * @param settings Settings providing the cloud parameters.
           * @param width Total output width in pixels (panels side by side).
           * @param height Output height in pixels.
           * @return RGBA8 image, or empty for a non-positive size.
           */
        C++: std::vector<unsigned char> (const RenderQualitySettings &, int, int)
        """

    @staticmethod
    def renderRainDebugRgba(settings: RenderQualitySettings, diagnostics: WeatherDiagnostics, width: int, height: int) -> list[int]:
        """
        /**
           * @brief Render rain/wet-material diagnostics into a four-panel RGBA8 image.
           *
           * The panels are precipitation density, wet-surface mask, puddle mask, and material override
           * strength. The CPU path uses the resolved weather diagnostics plus the same bounded
           * wet/puddle/ripple model used by the PBR shader, so tests and showcase images can inspect the
           * rain material path without relying on screenshots alone.
           * @param settings Settings providing the wet-material parameters.
           * @param diagnostics Resolved weather, e.g. weatherDiagnostics().
           * @param width Total output width in pixels (panels side by side).
           * @param height Output height in pixels.
           * @return RGBA8 image, or empty for a non-positive size.
           */
        C++: std::vector<unsigned char> (const RenderQualitySettings &, const WeatherDiagnostics &, int, int)
        """

    @staticmethod
    def renderSnowDebugRgba(settings: RenderQualitySettings, diagnostics: WeatherDiagnostics, width: int, height: int) -> list[int]:
        """
        /**
           * @brief Render snow-material diagnostics into a two-panel RGBA8 image.
           *
           * The panels are accumulation mask and material override strength. The CPU path uses resolved
           * weather diagnostics plus the same bounded snow accumulation model used by the PBR shader.
           * @param settings Settings providing the snow-material parameters.
           * @param diagnostics Resolved weather, e.g. weatherDiagnostics().
           * @param width Total output width in pixels (panels side by side).
           * @param height Output height in pixels.
           * @return RGBA8 image, or empty for a non-positive size.
           */
        C++: std::vector<unsigned char> (const RenderQualitySettings &, const WeatherDiagnostics &, int, int)
        """

    @staticmethod
    def transformRgbaForOutput(rgba: Sequence[int], width: int, height: int, transform: RayraiWindow.RgbaOutputTransform) -> list[int]:
        """
        /**
           * @brief Apply a bounded CPU output transform to RGBA8 pixels for screenshots/reports.
           *
           * This clamps RGB into displayable range and preserves alpha. It is intended for
           * explicit PNG/screenshot tooling, not for simulation sensor outputs unless requested.
           * @param rgba RGBA8 pixels.
           * @param width Image width in pixels.
           * @param height Image height in pixels.
           * @param transform Transform to apply (see RgbaOutputTransform for the order).
           * @return Transformed RGBA8 pixels, or empty for invalid input.
           */
        C++: std::vector<unsigned char> (const std::vector<unsigned char> &, int, int, const RgbaOutputTransform &)
        """

    @staticmethod
    def analyzeRgbaCalibrationPatches(rgba: Sequence[int], width: int, height: int, patches: Sequence[RayraiWindow.RgbaCalibrationPatch]) -> RayraiWindow.RgbaCalibrationMetrics:
        """
        /**
           * @brief Measure color-calibration patches in display-space RGBA8 pixels.
           *
           * Patch coordinates use top-left row order, matching RayRai PNG/debug buffers.
           * This is an offline/report helper for calibration scenes and does not mutate
           * renderer exposure, tone mapping, or sensor behavior.
           * @param rgba RGBA8 pixels; must contain at least width * height pixels.
           * @param width Image width in pixels.
           * @param height Image height in pixels.
           * @param patches Reference patches.
           * @return Per-patch and aggregate errors.
           */
        C++: RgbaCalibrationMetrics (const std::vector<unsigned char> &, int, int, const std::vector<RgbaCalibrationPatch> &)
        """

    @staticmethod
    def renderCalibrationPatchReportRgba(metrics: RayraiWindow.RgbaCalibrationMetrics, width: int, height: int) -> list[int]:
        """
        /**
           * @brief Render a compact calibration error report image from patch metrics.
           *
           * One row per patch: measured swatch, target swatch, and an error bar that
           * saturates at an RGB error of 0.5.
           * @param metrics Metrics from analyzeRgbaCalibrationPatches().
           * @param width Output width in pixels.
           * @param height Output height in pixels.
           * @return RGBA8 image, or empty for a non-positive size.
           */
        C++: std::vector<unsigned char> (const RgbaCalibrationMetrics &, int, int)
        """

    @staticmethod
    def recommendCalibrationOutputTransform(metrics: RayraiWindow.RgbaCalibrationMetrics) -> RayraiWindow.RgbaCalibrationRecommendation:
        """
        /**
           * @brief Build a bounded screenshot/report output transform from calibration patches.
           *
           * Neutral patches drive exposure and white balance; saturated patches drive a conservative
           * saturation correction. This returns a recommendation only and never mutates renderer state.
           * @param metrics Metrics from analyzeRgbaCalibrationPatches().
           * @return Recommendation; apply it with transformRgbaForOutput().
           */
        C++: RgbaCalibrationRecommendation (const RgbaCalibrationMetrics &)
        """

    def setDOFEnabled(self, e: bool) -> None:
        """
        /**
           * @brief Enable or disable depth of field post-processing.
           *
           * Overwritten by setRenderQualitySettings() (depthOfFieldEnabled).
           * @param e True to enable.
           */
        C++: void (bool)
        """

    def setDOFParams(self, focusDist: float, focusRange: float, maxRadius: float) -> None:
        """
        /**
           * @brief Set depth of field parameters.
           * @param focusDist Focus distance in view space.
           * @param focusRange Focus range around the focus distance.
           * @param maxRadius Max blur radius in pixels.
           *
           * Overwritten by setRenderQualitySettings().
           */
        C++: void (float, float, float)
        """

    def setShowCollisionBodies(self, show: bool) -> None:
        """
        /**
           * @brief Toggle rendering of collision shapes for articulated systems.
           * @param show True to render collision bodies (default false).
           */
        C++: void (bool)
        """

    def getShowCollisionBodies(self) -> bool:
        """
        /**
           * @brief Return collision-body rendering state.
           * @return True if collision bodies are shown.
           */
        C++: bool () const
        """

    def setHeightmapPatternResourcePath(self, path: str) -> None:
        """
        /**
           * @brief Set a tiled albedo texture for all heightmaps.
           * @param path Path to the texture image, or empty to clear.
           */
        C++: void (const std::string &)
        """

    def getHeightmapPatternResourcePath(self) -> str:
        """
        /// @return Path passed to setHeightmapPatternResourcePath(), empty when cleared.
        C++: std::string () const
        """

    def setHeightmapNormalResourcePath(self, path: str) -> None:
        """
        /// Set a tiled normal map (linear) for all heightmaps.
          /// @param path Path to the texture image, or empty to clear.
        C++: void (const std::string &)
        """

    def getHeightmapNormalResourcePath(self) -> str:
        """
        /// @return Path passed to setHeightmapNormalResourcePath(), empty when cleared.
        C++: std::string () const
        """

    def setHeightmapHeightResourcePath(self, path: str) -> None:
        """
        /// Set a tiled height map (linear) used for parallax on all heightmaps.
          /// @param path Path to the texture image, or empty to clear.
        C++: void (const std::string &)
        """

    def getHeightmapHeightResourcePath(self) -> str:
        """
        /// @return Path passed to setHeightmapHeightResourcePath(), empty when cleared.
        C++: std::string () const
        """

    def setHeightmapTerrainMaterialParameters(self, normalStrength: float, roughness: float, parallaxScale: float, deepParallax: bool = True) -> None:
        """
        /**
           * @brief Set heightmap terrain material parameters.
           * @param normalStrength Normal-map strength (default 1; negative acts as 0). Used only with a normal map.
           * @param roughness PBR roughness (default 0.9), clamped to [0.04, 1] when applied.
           * @param parallaxScale Parallax height scale (default 4). Used only with a height map.
           * @param deepParallax Use layered (deep) parallax mapping. Used only with a height map.
           */
        C++: void (float, float, float, bool)
        """

    def getHeightmapTerrainNormalStrength(self) -> float:
        """
        /// @return Normal strength set by setHeightmapTerrainMaterialParameters().
        C++: float () const
        """

    def getHeightmapTerrainRoughness(self) -> float:
        """
        /// @return Roughness set by setHeightmapTerrainMaterialParameters() (unclamped).
        C++: float () const
        """

    def getHeightmapTerrainDeepParallax(self) -> bool:
        """
        /// @return Deep-parallax flag set by setHeightmapTerrainMaterialParameters().
        C++: bool () const
        """

    def getHeightmapTerrainParallaxScale(self) -> float:
        """
        /// @return Parallax scale set by setHeightmapTerrainMaterialParameters().
        C++: float () const
        """

    def setHeightmapSplatLayer(self, albedoPath: str, controlTexId: int, primaryUvScale: float, secondaryUvScale: float, controlBounds: numpy.ndarray) -> None:
        """
        /**
           * @brief Configure a secondary terrain splat layer for heightmap meshes.
           *
           * The PBR mesh shader blends `albedoPath` over the primary heightmap albedo
           * using a single-channel control texture. Setting `controlTexId` to 0 or
           * `albedoPath` empty disables splat blending. Blending only happens when a
           * primary albedo (setHeightmapPatternResourcePath()) is set as well.
           *
           * @param albedoPath Secondary albedo image (sRGB); reloaded only when the path changes.
           * @param controlTexId Single-channel control texture (e.g. from updateSingleChannelTexture());
           *        its red channel is the secondary weight (0 = primary only, 1 = secondary only). Not owned.
           * All coordinates below are world XY of the terrain as if it were unrotated (its
           * translation applied, its orientation ignored), like raisim::HeightMap::getHeight();
           * for a rotated heightmap the layer therefore turns with the terrain.
           * @param primaryUvScale Primary albedo tiling: UV = world XY * scale (default 0.25).
           * @param secondaryUvScale Secondary albedo tiling: UV = world XY * scale (default 0.25).
           * @param controlBounds World XY rectangle covered by the control texture as
           *        (minX, minY, sizeX, sizeY) in metres; outside it the edge value is used.
           */
        C++: void (const std::string &, unsigned int, float, float, const glm::vec4 &)
        """

    def clearHeightmapSplatLayer(self) -> None:
        """
        /// Disable splat blending and forget the secondary albedo (the control texture is not deleted).
        C++: void ()
        """

    def getHeightmapSplatAlbedoId(self) -> int:
        """
        /// @return Secondary splat albedo texture id, 0 when unset.
        C++: unsigned int () const
        """

    def getHeightmapSplatControlId(self) -> int:
        """
        /// @return Splat control texture id, 0 when unset.
        C++: unsigned int () const
        """

    def getHeightmapSplatPrimaryUvScale(self) -> float:
        """
        /// @return Primary albedo UV scale used with splatting.
        C++: float () const
        """

    def getHeightmapSplatSecondaryUvScale(self) -> float:
        """
        /// @return Secondary albedo UV scale used with splatting.
        C++: float () const
        """

    def getHeightmapSplatControlBounds(self) -> numpy.ndarray:
        """
        /// @return Control texture bounds (minX, minY, sizeX, sizeY).
        C++: glm::vec4 () const
        """

    def hasHeightmapTerrainTextures(self) -> bool:
        """
        /// @return True when a heightmap albedo, normal or height texture, or a splat layer, is set.
        C++: bool () const
        """

    def applyHeightmapTerrainTextures(self, mesh: OpenGLMesh) -> None:
        """
        /**
           * @brief Give a heightmap mesh the terrain textures and material set above, as world
           * heightmaps get them. The mesh's texture coordinates must tile every 4 m of world XY,
           * like those of world heightmaps; the TCP viewer builds its heightmaps that way while
           * terrain textures are set.
           * @param mesh Heightmap mesh; its textures and material are replaced.
           */
        C++: void (OpenGLMesh &) const
        """

    def render(self, processEvents: bool = True) -> bool: ...

    def pollEvents(self) -> bool: ...

    def shouldClose(self) -> bool: ...

    def requestClose(self) -> None: ...

    def close(self) -> None: ...

    def swapWorld(self, world: raisimpy.World) -> None: ...

    def setTargetVisual(self, visual: Visuals | None = None) -> None: ...

    def invalidateExternalFrameCache(self, invalidateStaticSceneLayer: bool = True) -> None:
        """
        Invalidate cached frames. SDKs without layer-selective invalidation clear both layers.
        """

    def makeCurrent(self) -> None: ...

    def makeOffscreenContextCurrent(self) -> None: ...

    def detachOffscreenGlContext(self) -> None: ...

    @staticmethod
    def createOffscreenGlContext(world: raisimpy.World, width: int = 800, height: int = 600, title: str = 'rayrai_offscreen') -> RayraiWindow: ...

    @property
    def closed(self) -> bool: ...

    @property
    def world(self) -> raisimpy.World: ...

    @property
    def visible(self) -> bool: ...

    @property
    def size(self) -> tuple: ...

    def setVisible(self, visible: bool) -> None: ...

    def setTitle(self, title: str) -> None: ...

    def readRgba(self, flipVertical: bool = True) -> numpy.ndarray:
        """
        Return an owned uint8 image with shape (height, width, 4), after post-processing.
        """

    def saveScreenshot(self, path: str | os.PathLike) -> None: ...

    def __enter__(self) -> RayraiWindow: ...

    def __exit__(self, exc_type: object | None, exc_value: object | None, traceback: object | None) -> None: ...

    def focusOn(self, object: raisimpy.Object | None) -> None: ...

    def integrateWorldThreadSafe(self) -> None: ...

    def lockVisualizationServerMutex(self) -> None: ...

    def unlockVisualizationServerMutex(self) -> None: ...

    def addSlider(self, name: str, min: float, max: float, value: float) -> int: ...

    def getSlider(self, index: int) -> float: ...

    def getSliderValue(self, name: str) -> float: ...

    def setSliderValue(self, name: str, value: float) -> None: ...

    @staticmethod
    def updateSingleChannelTexture(existingTextureId: int, data: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)]) -> int: ...

    def getExternalFont(self) -> Font | None: ...

    def setExternalFont(self, font: Font | None = None) -> None: ...

    def loadFont(self, path: str, size: float = 16.0) -> Font: ...

    def debugObjectFor(self, object: raisimpy.Object) -> RaisimObject | None: ...

    def modelHandle(self, path: str) -> ModelHandle | None: ...

    def synchronousMeshLoading(self) -> RayraiWindow.SynchronousMeshLoadingScope: ...

    def setWeatherThunderCallback(self, callback: object | None) -> None: ...

Viewer: TypeAlias = RayraiWindow

RayraiViewer: TypeAlias = RayraiWindow

class OpenGLMesh:
    @overload
    def __init__(self, viewer: RayraiWindow, vertices: Sequence[Vertex], indices: Sequence[int], textures: Sequence[Texture] = [], baseColor: numpy.ndarray = ..., primitiveTopology: OpenGLMesh.PrimitiveTopology = OpenGLMesh.PrimitiveTopology.Triangles) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, vertices: Sequence[Vertex], indices: Sequence[int], textures: Sequence[Texture] = [], baseColor: numpy.ndarray = ..., primitiveTopology: OpenGLMesh.PrimitiveTopology = OpenGLMesh.PrimitiveTopology.Triangles) -> None: ...

    class GeometryDiagnostics:
        def __init__(self) -> None: ...

        @property
        def meshCount(self) -> int: ...

        @meshCount.setter
        def meshCount(self, arg: int, /) -> None: ...

        @property
        def triangleCount(self) -> int: ...

        @triangleCount.setter
        def triangleCount(self, arg: int, /) -> None: ...

        @property
        def tinyTriangleCount(self) -> int: ...

        @tinyTriangleCount.setter
        def tinyTriangleCount(self, arg: int, /) -> None: ...

        @property
        def minTriangleArea(self) -> float: ...

        @minTriangleArea.setter
        def minTriangleArea(self, arg: float, /) -> None: ...

        @property
        def maxTriangleArea(self) -> float: ...

        @maxTriangleArea.setter
        def maxTriangleArea(self, arg: float, /) -> None: ...

    class MeshDiagnostics:
        def __init__(self) -> None: ...

        @property
        def sourceMeshName(self) -> str: ...

        @sourceMeshName.setter
        def sourceMeshName(self, arg: str, /) -> None: ...

        @property
        def materialName(self) -> str: ...

        @materialName.setter
        def materialName(self, arg: str, /) -> None: ...

        @property
        def vertexCount(self) -> int: ...

        @vertexCount.setter
        def vertexCount(self, arg: int, /) -> None: ...

        @property
        def indexCount(self) -> int: ...

        @indexCount.setter
        def indexCount(self, arg: int, /) -> None: ...

        @property
        def triangleCount(self) -> int: ...

        @triangleCount.setter
        def triangleCount(self, arg: int, /) -> None: ...

        @property
        def tinyTriangleCount(self) -> int: ...

        @tinyTriangleCount.setter
        def tinyTriangleCount(self, arg: int, /) -> None: ...

        @property
        def minTriangleArea(self) -> float: ...

        @minTriangleArea.setter
        def minTriangleArea(self, arg: float, /) -> None: ...

        @property
        def maxTriangleArea(self) -> float: ...

        @maxTriangleArea.setter
        def maxTriangleArea(self, arg: float, /) -> None: ...

        @property
        def materialTextureReferenceCount(self) -> int: ...

        @materialTextureReferenceCount.setter
        def materialTextureReferenceCount(self, arg: int, /) -> None: ...

        @property
        def textureSlots(self) -> list[str]: ...

        @textureSlots.setter
        def textureSlots(self, arg: Sequence[str], /) -> None: ...

        @property
        def perTextureUvTransformCount(self) -> int: ...

        @perTextureUvTransformCount.setter
        def perTextureUvTransformCount(self, arg: int, /) -> None: ...

        @property
        def uvTransformSlots(self) -> list[str]: ...

        @uvTransformSlots.setter
        def uvTransformSlots(self, arg: Sequence[str], /) -> None: ...

        @property
        def shaderVariant(self) -> str: ...

        @shaderVariant.setter
        def shaderVariant(self, arg: str, /) -> None: ...

        @property
        def alphaMode(self) -> str: ...

        @alphaMode.setter
        def alphaMode(self, arg: str, /) -> None: ...

        @property
        def blendMode(self) -> str: ...

        @blendMode.setter
        def blendMode(self, arg: str, /) -> None: ...

        @property
        def renderPriority(self) -> int: ...

        @renderPriority.setter
        def renderPriority(self, arg: int, /) -> None: ...

        @property
        def hasNextPass(self) -> bool: ...

        @hasNextPass.setter
        def hasNextPass(self, arg: bool, /) -> None: ...

        @property
        def nextPassCount(self) -> int: ...

        @nextPassCount.setter
        def nextPassCount(self, arg: int, /) -> None: ...

        @property
        def nextPassMaterials(self) -> list[str]: ...

        @nextPassMaterials.setter
        def nextPassMaterials(self, arg: Sequence[str], /) -> None: ...

        @property
        def detailBlendMode(self) -> str: ...

        @detailBlendMode.setter
        def detailBlendMode(self, arg: str, /) -> None: ...

        @property
        def cullMode(self) -> str: ...

        @cullMode.setter
        def cullMode(self, arg: str, /) -> None: ...

        @property
        def textureRepeatMode(self) -> str: ...

        @textureRepeatMode.setter
        def textureRepeatMode(self, arg: str, /) -> None: ...

        @property
        def textureFilter(self) -> str: ...

        @textureFilter.setter
        def textureFilter(self, arg: str, /) -> None: ...

        @property
        def albedoTextureForceSrgb(self) -> bool: ...

        @albedoTextureForceSrgb.setter
        def albedoTextureForceSrgb(self, arg: bool, /) -> None: ...

        @property
        def albedoTextureMsdf(self) -> bool: ...

        @albedoTextureMsdf.setter
        def albedoTextureMsdf(self, arg: bool, /) -> None: ...

        @property
        def msdfPixelRange(self) -> float: ...

        @msdfPixelRange.setter
        def msdfPixelRange(self, arg: float, /) -> None: ...

        @property
        def msdfOutlineSize(self) -> float: ...

        @msdfOutlineSize.setter
        def msdfOutlineSize(self, arg: float, /) -> None: ...

        @property
        def emissionOperator(self) -> str: ...

        @emissionOperator.setter
        def emissionOperator(self, arg: str, /) -> None: ...

        @property
        def heightmapInvert(self) -> bool: ...

        @heightmapInvert.setter
        def heightmapInvert(self, arg: bool, /) -> None: ...

        @property
        def heightmapVertexDisplacementScale(self) -> float: ...

        @heightmapVertexDisplacementScale.setter
        def heightmapVertexDisplacementScale(self, arg: float, /) -> None: ...

        @property
        def heightmapVertexDisplacementMidlevel(self) -> float: ...

        @heightmapVertexDisplacementMidlevel.setter
        def heightmapVertexDisplacementMidlevel(self, arg: float, /) -> None: ...

        @property
        def billboardMode(self) -> str: ...

        @billboardMode.setter
        def billboardMode(self, arg: str, /) -> None: ...

        @property
        def billboardKeepScale(self) -> bool: ...

        @billboardKeepScale.setter
        def billboardKeepScale(self, arg: bool, /) -> None: ...

        @property
        def particlesAnimHFrames(self) -> int: ...

        @particlesAnimHFrames.setter
        def particlesAnimHFrames(self, arg: int, /) -> None: ...

        @property
        def particlesAnimVFrames(self) -> int: ...

        @particlesAnimVFrames.setter
        def particlesAnimVFrames(self, arg: int, /) -> None: ...

        @property
        def particlesAnimLoop(self) -> bool: ...

        @particlesAnimLoop.setter
        def particlesAnimLoop(self, arg: bool, /) -> None: ...

        @property
        def particlesAnimFrameProgress(self) -> float: ...

        @particlesAnimFrameProgress.setter
        def particlesAnimFrameProgress(self, arg: float, /) -> None: ...

        @property
        def particleTrailsEnabled(self) -> bool: ...

        @particleTrailsEnabled.setter
        def particleTrailsEnabled(self, arg: bool, /) -> None: ...

        @property
        def particleTrailLength(self) -> float: ...

        @particleTrailLength.setter
        def particleTrailLength(self, arg: float, /) -> None: ...

        @property
        def particleTrailTaper(self) -> float: ...

        @particleTrailTaper.setter
        def particleTrailTaper(self, arg: float, /) -> None: ...

        @property
        def particleTrailWorldDirection(self) -> numpy.ndarray: ...

        @particleTrailWorldDirection.setter
        def particleTrailWorldDirection(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def fixedSize(self) -> bool: ...

        @fixedSize.setter
        def fixedSize(self, arg: bool, /) -> None: ...

        @property
        def usePointSize(self) -> bool: ...

        @usePointSize.setter
        def usePointSize(self, arg: bool, /) -> None: ...

        @property
        def pointSize(self) -> float: ...

        @pointSize.setter
        def pointSize(self, arg: float, /) -> None: ...

        @property
        def useZClipScale(self) -> bool: ...

        @useZClipScale.setter
        def useZClipScale(self, arg: bool, /) -> None: ...

        @property
        def zClipScale(self) -> float: ...

        @zClipScale.setter
        def zClipScale(self, arg: float, /) -> None: ...

        @property
        def useFovOverride(self) -> bool: ...

        @useFovOverride.setter
        def useFovOverride(self, arg: bool, /) -> None: ...

        @property
        def fovOverrideDegrees(self) -> float: ...

        @fovOverrideDegrees.setter
        def fovOverrideDegrees(self, arg: float, /) -> None: ...

        @property
        def secondaryUvCount(self) -> int: ...

        @secondaryUvCount.setter
        def secondaryUvCount(self, arg: int, /) -> None: ...

        @property
        def detailUsesUv2(self) -> bool: ...

        @detailUsesUv2.setter
        def detailUsesUv2(self, arg: bool, /) -> None: ...

        @property
        def aoUsesUv2(self) -> bool: ...

        @aoUsesUv2.setter
        def aoUsesUv2(self, arg: bool, /) -> None: ...

        @property
        def emissiveUsesUv2(self) -> bool: ...

        @emissiveUsesUv2.setter
        def emissiveUsesUv2(self, arg: bool, /) -> None: ...

        @property
        def lightmapUsesUv2(self) -> bool: ...

        @lightmapUsesUv2.setter
        def lightmapUsesUv2(self, arg: bool, /) -> None: ...

        @property
        def uv2TriplanarEnabled(self) -> bool: ...

        @uv2TriplanarEnabled.setter
        def uv2TriplanarEnabled(self, arg: bool, /) -> None: ...

        @property
        def uv2TriplanarWorldSpace(self) -> bool: ...

        @uv2TriplanarWorldSpace.setter
        def uv2TriplanarWorldSpace(self, arg: bool, /) -> None: ...

        @property
        def uv2TriplanarSharpness(self) -> float: ...

        @uv2TriplanarSharpness.setter
        def uv2TriplanarSharpness(self, arg: float, /) -> None: ...

        @property
        def vertexColorCount(self) -> int: ...

        @vertexColorCount.setter
        def vertexColorCount(self, arg: int, /) -> None: ...

        @property
        def vertexColorAsAlbedo(self) -> bool: ...

        @vertexColorAsAlbedo.setter
        def vertexColorAsAlbedo(self, arg: bool, /) -> None: ...

        @property
        def doubleSided(self) -> bool: ...

        @doubleSided.setter
        def doubleSided(self, arg: bool, /) -> None: ...

        @property
        def unlit(self) -> bool: ...

        @unlit.setter
        def unlit(self, arg: bool, /) -> None: ...

        @property
        def normalMapped(self) -> bool: ...

        @normalMapped.setter
        def normalMapped(self, arg: bool, /) -> None: ...

        @property
        def tangentSource(self) -> str: ...

        @tangentSource.setter
        def tangentSource(self, arg: str, /) -> None: ...

        @property
        def validTangentCount(self) -> int: ...

        @validTangentCount.setter
        def validTangentCount(self, arg: int, /) -> None: ...

        @property
        def fallbackTangentCount(self) -> int: ...

        @fallbackTangentCount.setter
        def fallbackTangentCount(self, arg: int, /) -> None: ...

        @property
        def generatedTangents(self) -> bool: ...

        @generatedTangents.setter
        def generatedTangents(self, arg: bool, /) -> None: ...

        @property
        def importedTangents(self) -> bool: ...

        @importedTangents.setter
        def importedTangents(self, arg: bool, /) -> None: ...

        @property
        def decal(self) -> bool: ...

        @decal.setter
        def decal(self, arg: bool, /) -> None: ...

        @property
        def tinySurfaceDetail(self) -> bool: ...

        @tinySurfaceDetail.setter
        def tinySurfaceDetail(self, arg: bool, /) -> None: ...

        @property
        def transmission(self) -> bool: ...

        @transmission.setter
        def transmission(self, arg: bool, /) -> None: ...

        @property
        def refraction(self) -> bool: ...

        @refraction.setter
        def refraction(self, arg: bool, /) -> None: ...

        @property
        def volume(self) -> bool: ...

        @volume.setter
        def volume(self, arg: bool, /) -> None: ...

        @property
        def subsurfaceSkinMode(self) -> bool: ...

        @subsurfaceSkinMode.setter
        def subsurfaceSkinMode(self, arg: bool, /) -> None: ...

        @property
        def clearcoat(self) -> bool: ...

        @clearcoat.setter
        def clearcoat(self, arg: bool, /) -> None: ...

        @property
        def sheen(self) -> bool: ...

        @sheen.setter
        def sheen(self, arg: bool, /) -> None: ...

        @property
        def anisotropy(self) -> bool: ...

        @anisotropy.setter
        def anisotropy(self, arg: bool, /) -> None: ...

        @property
        def specularIor(self) -> bool: ...

        @specularIor.setter
        def specularIor(self, arg: bool, /) -> None: ...

        @property
        def planarReflection(self) -> bool: ...

        @planarReflection.setter
        def planarReflection(self, arg: bool, /) -> None: ...

        @property
        def environmentIbl(self) -> bool: ...

        @environmentIbl.setter
        def environmentIbl(self, arg: bool, /) -> None: ...

        @property
        def materialRenderFlags(self) -> bool: ...

        @materialRenderFlags.setter
        def materialRenderFlags(self, arg: bool, /) -> None: ...

        @property
        def thinTwoSidedLighting(self) -> bool: ...

        @thinTwoSidedLighting.setter
        def thinTwoSidedLighting(self, arg: bool, /) -> None: ...

        @property
        def thinTransmission(self) -> bool: ...

        @thinTransmission.setter
        def thinTransmission(self, arg: bool, /) -> None: ...

        @property
        def subsurface(self) -> bool: ...

        @subsurface.setter
        def subsurface(self, arg: bool, /) -> None: ...

        @property
        def backlight(self) -> bool: ...

        @backlight.setter
        def backlight(self, arg: bool, /) -> None: ...

        @property
        def foliage(self) -> bool: ...

        @foliage.setter
        def foliage(self, arg: bool, /) -> None: ...

        @property
        def foliageTwoSidedLighting(self) -> bool: ...

        @foliageTwoSidedLighting.setter
        def foliageTwoSidedLighting(self, arg: bool, /) -> None: ...

        @property
        def foliageTransmission(self) -> bool: ...

        @foliageTransmission.setter
        def foliageTransmission(self, arg: bool, /) -> None: ...

        @property
        def foliageWeatherResponse(self) -> bool: ...

        @foliageWeatherResponse.setter
        def foliageWeatherResponse(self, arg: bool, /) -> None: ...

        @property
        def weatherAffectsWetness(self) -> bool: ...

        @weatherAffectsWetness.setter
        def weatherAffectsWetness(self, arg: bool, /) -> None: ...

        @property
        def weatherAffectsSnow(self) -> bool: ...

        @weatherAffectsSnow.setter
        def weatherAffectsSnow(self, arg: bool, /) -> None: ...

        @property
        def weatherPuddleEligible(self) -> bool: ...

        @weatherPuddleEligible.setter
        def weatherPuddleEligible(self, arg: bool, /) -> None: ...

        @property
        def weatherSnowEligible(self) -> bool: ...

        @weatherSnowEligible.setter
        def weatherSnowEligible(self, arg: bool, /) -> None: ...

        @property
        def weatherMaskMapped(self) -> bool: ...

        @weatherMaskMapped.setter
        def weatherMaskMapped(self, arg: bool, /) -> None: ...

        @property
        def weatherPorosity(self) -> float: ...

        @weatherPorosity.setter
        def weatherPorosity(self, arg: float, /) -> None: ...

        @property
        def weatherSkipReason(self) -> str: ...

        @weatherSkipReason.setter
        def weatherSkipReason(self, arg: str, /) -> None: ...

        @property
        def depthTestMode(self) -> str: ...

        @depthTestMode.setter
        def depthTestMode(self, arg: str, /) -> None: ...

        @property
        def depthTestOverride(self) -> bool: ...

        @depthTestOverride.setter
        def depthTestOverride(self, arg: bool, /) -> None: ...

        @property
        def depthWriteOverride(self) -> bool: ...

        @depthWriteOverride.setter
        def depthWriteOverride(self, arg: bool, /) -> None: ...

        @property
        def depthTestResolved(self) -> bool: ...

        @depthTestResolved.setter
        def depthTestResolved(self, arg: bool, /) -> None: ...

        @property
        def depthWriteResolved(self) -> bool: ...

        @depthWriteResolved.setter
        def depthWriteResolved(self, arg: bool, /) -> None: ...

        @property
        def stencilOverride(self) -> bool: ...

        @stencilOverride.setter
        def stencilOverride(self, arg: bool, /) -> None: ...

        @property
        def stencilRead(self) -> bool: ...

        @stencilRead.setter
        def stencilRead(self, arg: bool, /) -> None: ...

        @property
        def stencilWrite(self) -> bool: ...

        @stencilWrite.setter
        def stencilWrite(self, arg: bool, /) -> None: ...

        @property
        def stencilWriteDepthFail(self) -> bool: ...

        @stencilWriteDepthFail.setter
        def stencilWriteDepthFail(self, arg: bool, /) -> None: ...

        @property
        def stencilCompare(self) -> str: ...

        @stencilCompare.setter
        def stencilCompare(self, arg: str, /) -> None: ...

        @property
        def stencilReference(self) -> int: ...

        @stencilReference.setter
        def stencilReference(self, arg: int, /) -> None: ...

        @property
        def stencilMask(self) -> int: ...

        @stencilMask.setter
        def stencilMask(self, arg: int, /) -> None: ...

        @property
        def stencilEffectMode(self) -> str: ...

        @stencilEffectMode.setter
        def stencilEffectMode(self, arg: str, /) -> None: ...

        @property
        def stencilEffectColor(self) -> numpy.ndarray: ...

        @stencilEffectColor.setter
        def stencilEffectColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def stencilOutlineThickness(self) -> float: ...

        @stencilOutlineThickness.setter
        def stencilOutlineThickness(self, arg: float, /) -> None: ...

        @property
        def pbr(self) -> bool: ...

        @pbr.setter
        def pbr(self, arg: bool, /) -> None: ...

    class PreparedAttributes:
        def __init__(self) -> None: ...

        @property
        def tangentSource(self) -> str: ...

        @tangentSource.setter
        def tangentSource(self, arg: str, /) -> None: ...

        @property
        def validTangentCount(self) -> int: ...

        @validTangentCount.setter
        def validTangentCount(self, arg: int, /) -> None: ...

        @property
        def fallbackTangentCount(self) -> int: ...

        @fallbackTangentCount.setter
        def fallbackTangentCount(self, arg: int, /) -> None: ...

        @property
        def uvDensity(self) -> numpy.ndarray: ...

        @uvDensity.setter
        def uvDensity(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def uvEdgeLength(self) -> numpy.ndarray: ...

        @uvEdgeLength.setter
        def uvEdgeLength(self, arg: numpy.ndarray, /) -> None: ...

    class PrimitiveTopology(enum.Enum):
        Triangles = 0

        Lines = 1

        Points = 2

    class RenderMode(enum.Enum):
        All = 0

        SimpleOnly = 1

        PbrOnly = 2

        PbrNonTransmissiveOnly = 3

        PbrTransmissiveOnly = 4

    class StagedGeometry:
        def __init__(self) -> None: ...

        @property
        def vertexBuffer(self) -> int: ...

        @vertexBuffer.setter
        def vertexBuffer(self, arg: int, /) -> None: ...

        @property
        def vertexOffset(self) -> int: ...

        @vertexOffset.setter
        def vertexOffset(self, arg: int, /) -> None: ...

        @property
        def indexBuffer(self) -> int: ...

        @indexBuffer.setter
        def indexBuffer(self, arg: int, /) -> None: ...

        @property
        def indexOffset(self) -> int: ...

        @indexOffset.setter
        def indexOffset(self, arg: int, /) -> None: ...

    class VertexSummaries:
        def __init__(self) -> None: ...

        @property
        def bounds(self) -> bool: ...

        @bounds.setter
        def bounds(self, arg: bool, /) -> None: ...

        @property
        def sphere(self) -> numpy.ndarray: ...

        @sphere.setter
        def sphere(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def halfExtents(self) -> numpy.ndarray: ...

        @halfExtents.setter
        def halfExtents(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def uniformColor(self) -> bool: ...

        @uniformColor.setter
        def uniformColor(self, arg: bool, /) -> None: ...

        @property
        def contentDigest(self) -> int: ...

        @contentDigest.setter
        def contentDigest(self, arg: int, /) -> None: ...

        @property
        def maxIndex(self) -> int: ...

        @maxIndex.setter
        def maxIndex(self, arg: int, /) -> None: ...

    def drawsInMode(self, mode: OpenGLMesh.RenderMode, materialOverride: Material | None) -> bool:
        """
        /**
           * Whether draw() with the same arguments would rasterize anything. It mirrors
           * the mode and transmission filters draw() applies, including the nextPass
           * chain, so a caller can skip a program bind and a whole submission instead
           * of paying for a draw that returns immediately.
           * @param mode Material filter.
           * @param materialOverride Material to test instead of #material, or nullptr.
           * @return True if draw() would issue at least one draw call.
           */
        C++: bool (RenderMode, const raisin::Material *) const
        """

    @staticmethod
    def isPbrRenderMode(mode: OpenGLMesh.RenderMode) -> bool:
        """
        /// @param mode Render mode to test.
          /// @return True for the three PBR-only render modes.
        C++: bool (RenderMode)
        """

    @property
    def vertices(self) -> list[Vertex]: ...

    @vertices.setter
    def vertices(self, arg: Sequence[Vertex], /) -> None: ...

    @property
    def indices(self) -> list[int]: ...

    @indices.setter
    def indices(self, arg: Sequence[int], /) -> None: ...

    @property
    def textures(self) -> list[Texture]: ...

    @textures.setter
    def textures(self, arg: Sequence[Texture], /) -> None: ...

    @property
    def baseColor(self) -> numpy.ndarray: ...

    @baseColor.setter
    def baseColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def material(self) -> Material: ...

    @material.setter
    def material(self, arg: Material, /) -> None: ...

    @property
    def debugSourceMeshName(self) -> str: ...

    @debugSourceMeshName.setter
    def debugSourceMeshName(self, arg: str, /) -> None: ...

    @property
    def debugImportedFaceCount(self) -> int: ...

    @debugImportedFaceCount.setter
    def debugImportedFaceCount(self, arg: int, /) -> None: ...

    @property
    def debugImportedIndexCount(self) -> int: ...

    @debugImportedIndexCount.setter
    def debugImportedIndexCount(self, arg: int, /) -> None: ...

    @property
    def debugGeneratedLodMaxUvClusterSpan(self) -> float: ...

    @debugGeneratedLodMaxUvClusterSpan.setter
    def debugGeneratedLodMaxUvClusterSpan(self, arg: float, /) -> None: ...

    @property
    def generatedLodRelativeError(self) -> float: ...

    @generatedLodRelativeError.setter
    def generatedLodRelativeError(self, arg: float, /) -> None: ...

    @property
    def VAO(self) -> int: ...

    @VAO.setter
    def VAO(self, arg: int, /) -> None: ...

    @staticmethod
    def createIndexedRangeView(owner: OpenGLMesh, localVertices: Sequence[Vertex], localIndices: Sequence[int], firstIndex: int, baseVertex: int, viewMaterial: Material, viewBaseColor: numpy.ndarray, debugName: str = '') -> OpenGLMesh:
        """
        /**
           * @brief Create a CPU-metadata view into @p owner's indexed GPU buffers.
           *
           * The returned mesh has its own material/CPU geometry metadata but shares
           * @p owner's VAO/EBO/VBO and draws the requested index range. It does not own
           * or delete the shared GL handles, and holds a `shared_ptr` to @p owner so the
           * borrowed GL handles cannot dangle.
           *
           * Taking the owner as an explicit `std::shared_ptr` makes shared ownership a
           * compile-time requirement (no `shared_from_this`/`bad_weak_ptr` fallback that
           * could silently return a non-owning view), and the view is built without any
           * throwaway GL upload — it never allocates its own VAO/VBO/EBO. Returns null if
           * @p owner is null.
           * @param owner Mesh whose GPU buffers are borrowed.
           * @param localVertices CPU vertices for this view (bounds/diagnostics only).
           * @param localIndices CPU indices for this view; their count is the draw count.
           * @param firstIndex Offset, in indices, into @p owner's index buffer.
           * @param baseVertex Value added to each index when drawing.
           * @param viewMaterial Material of the view.
           * @param viewBaseColor Base color of the view.
           * @param debugName Diagnostics name; empty inherits @p owner's.
           * @return The view, or nullptr.
           */
        C++: std::shared_ptr<OpenGLMesh> (std::shared_ptr<const OpenGLMesh>, const std::vector<Vertex> &, const std::vector<unsigned int> &, size_t, int, const raisin::Material &, const glm::vec4 &, const std::string &)
        """

    @staticmethod
    def createSharedGpuGeometryView(owner: OpenGLMesh) -> OpenGLMesh:
        """
        /**
           * @brief Create a context-local mesh wrapper that reuses another mesh's GPU buffers.
           *
           * Buffer objects and textures are shared by OpenGL context share groups, while
           * vertex array objects are context-local. This factory copies the CPU/material
           * metadata, retains the shared VBO/EBO allocation, and creates a VAO in the
           * caller's current context.
           * @param owner Source mesh; its buffers must be visible in the current context.
           * @return The view, or nullptr if @p owner is null, has no shareable buffers,
           *         or is not visible from the current context.
           */
        C++: std::shared_ptr<OpenGLMesh> (const std::shared_ptr<const OpenGLMesh> &)
        """

    def sharedGpuAssetSnapshot(self) -> OpenGLMesh:
        """
        /// Freeze metadata before publishing to the asset cache. The template owns
          /// no VAO and remains immutable while worker-local materials are edited.
          /// Internal. @return The cached snapshot, or nullptr for meshes that reference any
          /// texture or a material next-pass (those are never shared across contexts).
        C++: std::shared_ptr<const OpenGLMesh> ()
        """

    def gpuGeometryVisibleInCurrentContext(self) -> bool:
        """
        /**
           * @return True when this mesh's VBO/EBO are visible in the current GL context.
           */
        C++: bool () const
        """

    def gpuGeometryBytes(self) -> int:
        """
        /**
           * @return Requested VBO + EBO allocation size for this mesh, in bytes.
           */
        C++: size_t () const
        """

    @overload
    @staticmethod
    def createCpuGeometry(vertices: Sequence[Vertex], indices: Sequence[int], textures: Sequence[Texture], baseColor: numpy.ndarray, prepareAttributes: bool = True) -> OpenGLMesh:
        """
        /// Prepare geometry on a worker without issuing any OpenGL calls.
          /// Call uploadPreparedGeometry() on the GL thread before drawing.
          /// @param vertices Vertex array.
          /// @param indices Triangle index array.
          /// @param textures Texture list (ids are only recorded).
          /// @param baseColor Base color.
          /// @param prepareAttributes Also run prepareGeometryAttributes().
          /// @return New mesh with no GL objects.
        C++: std::shared_ptr<OpenGLMesh> (const std::vector<Vertex> &, const std::vector<unsigned int> &, const std::vector<Texture> &, const glm::vec4 &, bool)
        """

    @overload
    @staticmethod
    def createCpuGeometry(vertices: Sequence[Vertex], indices: Sequence[int], textures: Sequence[Texture], baseColor: numpy.ndarray, prepareAttributes: bool = True) -> OpenGLMesh:
        """
        /// createCpuGeometry() that takes over the vertex and index arrays instead of copying them.
        C++: std::shared_ptr<OpenGLMesh> (std::vector<Vertex> &&, std::vector<unsigned int> &&, const std::vector<Texture> &, const glm::vec4 &, bool)
        """

    def prepareGeometryAttributes(self) -> None:
        """
        /// Compute missing tangents and UV statistics without a GL context.
        C++: void ()
        """

    def preparedAttributes(self) -> OpenGLMesh.PreparedAttributes:
        """
        /// @return The attributes the last prepareGeometryAttributes() derived.
        C++: PreparedAttributes () const
        """

    def restorePreparedGeometry(self, vertices: Sequence[Vertex], indices: Sequence[int], attributes: OpenGLMesh.PreparedAttributes) -> None:
        """
        /// Replace the CPU geometry with an already prepared copy, as
          /// prepareGeometryAttributes() would have left it. Internal: no GL calls;
          /// for caches of prepared meshes.
          /// @param vertices Prepared vertex array.
          /// @param indices Triangle index array.
          /// @param attributes The attributes prepared with them.
        C++: void (std::vector<Vertex> &&, std::vector<unsigned int> &&, const PreparedAttributes &)
        """

    def precomputeVertexSummaries(self) -> None:
        """
        /// Compute what the first draws would otherwise scan #vertices for: the
          /// local bounds and whether every vertex has the same color. Internal: no
          /// GL calls; workers call it before the render thread uses the mesh.
        C++: void () const
        """

    def vertexSummaries(self) -> OpenGLMesh.VertexSummaries:
        """
        /// @return The summaries of the current vertices and indices.
        C++: VertexSummaries () const
        """

    def adoptVertexSummaries(self, summaries: OpenGLMesh.VertexSummaries) -> None:
        """
        /// Adopt summaries a cache stored for exactly these vertices and indices.
        C++: void (const VertexSummaries &) const
        """

    def contentDigest(self) -> int:
        """
        /// Hash of #vertices and #indices; for summaries adopted from a cache of
          /// prepared geometry (until a method of this class changes the geometry),
          /// the digest stored with them. Internal: keys caches derived from the geometry.
        C++: uint64_t () const
        """

    @overload
    def uploadPreparedGeometry(self, compactIndices: bool = True) -> None:
        """
        /// Upload an already-prepared CPU mesh on the context-owning thread.
          /// No-op if the mesh already has a VAO or borrows another mesh's buffers.
          /// @param compactIndices Allow 16-bit GPU indices when lossless.
        C++: void (bool)
        """

    @overload
    def uploadPreparedGeometry(self, compactIndices: bool, staged: OpenGLMesh.StagedGeometry) -> None:
        """
        /// uploadPreparedGeometry() that copies the geometry on the GPU, from a staging buffer.
        C++: void (bool, const StagedGeometry &)
        """

    def uploadsCompactIndices(self, compactIndices: bool) -> bool:
        """
        /// @return Whether an upload allowed `compactIndices` stores 16-bit indices
          ///         (triangles whose indices all fit).
        C++: bool (bool) const
        """

    @overload
    def draw(self, shader: Shader) -> None:
        """
        /**
           * @brief Draw the mesh using the provided shader.
           *
           * Same as `draw(shader, RenderMode::All)`. The shader is not bound here and
           * must already be in use; the model matrix and `objectColor` are the caller's
           * responsibility. Material depth/stencil/blend/cull state is applied around
           * the draw call.
           * @param shader Shader with expected texture uniforms.
           */
        C++: void (const Shader &) const
        """

    @overload
    def draw(self, shader: Shader, mode: OpenGLMesh.RenderMode, materialOverride: Material | None = None) -> None:
        """
        /**
           * @brief Draw the mesh if it matches the requested material render mode.
           *
           * Non-PBR modes set the simple-shader material uniforms and bind the diffuse,
           * specular and normal maps to units 0, 1 and 2 (white fallback when absent).
           * PBR modes apply the full material (Material::applyTo()) and also draw its
           * next-pass chain. The shader must already be in use.
           * @param shader Shader with expected material uniforms.
           * @param mode Material filter for multi-pass rendering.
           * @param materialOverride Optional material to use instead of the mesh material
           *        (the mesh's #textures are then ignored).
           */
        C++: void (const Shader &, RenderMode, const raisin::Material *) const
        """

    def drawInstanced(self, shader: Shader, instanceCount: int) -> None:
        """
        /**
           * @brief Draw the mesh with instancing using the mesh's own VAO.
           *
           * Per-instance attributes must already be configured; usually prefer
           * drawInstancedWithVao() with a VAO from instancedVaoForBuffer().
           * @param shader Shader with expected texture uniforms.
           * @param instanceCount Number of instances to draw; <= 0 draws nothing.
           */
        C++: void (const Shader &, int) const
        """

    def drawInstancedWithVao(self, shader: Shader, instanceCount: int, vao: int) -> None:
        """
        /**
           * @brief Draw the mesh with instancing using an explicit VAO variant.
           * @param shader Shader with expected texture uniforms.
           * @param instanceCount Number of instances to draw.
           * @param vao VAO from instancedVaoForBuffer(), or 0 for the mesh's own VAO.
           */
        C++: void (const Shader &, int, unsigned int) const
        """

    def drawInstancedIndirect(self, shader: Shader, indirectBuffer: int) -> None:
        """
        /**
           * @brief Draw the mesh with an OpenGL DrawElementsIndirect command buffer.
           * @param shader Shader with expected texture uniforms.
           * @param indirectBuffer Buffer containing a DrawElementsIndirectCommand.
           */
        C++: void (const Shader &, unsigned int) const
        """

    def drawInstancedIndirectWithVao(self, shader: Shader, indirectBuffer: int, vao: int, commandOffsetBytes: int = 0) -> None:
        """
        /**
           * @brief Draw an indirect instanced command using an explicit VAO variant.
           * @param shader Shader with expected texture uniforms.
           * @param indirectBuffer Buffer containing a DrawElementsIndirectCommand; 0 draws nothing.
           * @param vao VAO from instancedVaoForBuffer(), or 0 for the mesh's own VAO.
           * @param commandOffsetBytes Byte offset of the command in @p indirectBuffer.
           */
        C++: void (const Shader &, unsigned int, unsigned int, size_t) const
        """

    def drawInstancedMultiIndirect(self, shader: Shader, indirectBuffer: int, commandCount: int, commandStrideBytes: int) -> None:
        """
        /**
           * @brief Draw several compatible DrawElementsIndirect commands against this mesh's VAO.
           * @param shader Shader with expected texture uniforms.
           * @param indirectBuffer Buffer containing tightly packed indirect commands.
           * @param commandCount Number of commands in @p indirectBuffer.
           * @param commandStrideBytes Stride between commands, or zero for tightly packed commands.
           */
        C++: void (const Shader &, unsigned int, int, int) const
        """

    def drawInstancedMultiIndirectWithVao(self, shader: Shader, indirectBuffer: int, commandCount: int, commandStrideBytes: int, vao: int) -> None:
        """
        /**
           * @brief Draw several indirect commands using an explicit VAO variant.
           * @param shader Shader with expected texture uniforms.
           * @param indirectBuffer Buffer containing the indirect commands.
           * @param commandCount Number of commands; <= 0 draws nothing.
           * @param commandStrideBytes Stride between commands, or zero for tightly packed commands.
           * @param vao VAO from instancedVaoForBuffer(), or 0 for the mesh's own VAO.
           */
        C++: void (const Shader &, unsigned int, int, int, unsigned int) const
        """

    @overload
    def instancedVaoForBuffer(self, instanceBuffer: int) -> int:
        """
        /**
           * @brief Return a VAO whose base mesh attributes plus instance attributes use @p instanceBuffer.
           *
           * The instance buffer holds one 80-byte record per instance: a column-major
           * `mat4` model matrix (attributes 3-6) followed by a `vec4` color
           * (attribute 7). VAOs are created on first use and cached per buffer; call
           * releaseInstancedVaoForBuffer() before deleting the buffer.
           * @param instanceBuffer GL array buffer with instance records; 0 returns 0.
           * @return VAO id owned by this mesh, or 0 on failure.
           */
        C++: unsigned int (unsigned int) const
        """

    @overload
    def instancedVaoForBuffer(self, instanceBuffer: int, compactFoliage: bool) -> int:
        """
        /**
           * @brief Internal: as above, optionally with the compact foliage vertex layout.
           * @param instanceBuffer GL array buffer with instance records.
           * @param compactFoliage Use the packed foliage vertex stream when available.
           * @return VAO id owned by this mesh, or 0 on failure.
           */
        C++: unsigned int (unsigned int, bool) const
        """

    def releaseInstancedVaoForBuffer(self, instanceBuffer: int) -> None:
        """
        /**
           * @brief Drop cached VAO variants that reference @p instanceBuffer.
           * @param instanceBuffer Instance buffer about to be deleted or replaced.
           */
        C++: void (unsigned int) const
        """

    def drawWithBoundShader(self, shader: Shader) -> None:
        """
        /**
           * @brief Same as draw(), but assumes the caller has already bound @p shader.
           *        Currently equivalent to draw() (kept as an API hook for callers that
           *        used to pay a redundant shader bind cost; preserved for callers that
           *        opt in via OpenGLMesh::drawWithBoundShader).
           * @param shader Shader that is already in use.
           */
        C++: void (const Shader &) const
        """

    def drawInstancedWithBoundShader(self, shader: Shader, instanceCount: int) -> None:
        """
        /// Equivalent to drawInstanced(); kept for API parity with drawWithBoundShader().
          /// @param shader Shader that is already in use.
          /// @param instanceCount Number of instances to draw.
        C++: void (const Shader &, int) const
        """

    def drawIndexCount(self) -> int:
        """
        /// @return Number of indices issued per draw (the view range for range views).
        C++: size_t () const
        """

    @staticmethod
    def createWithCompactIndices(vertices: Sequence[Vertex], indices: Sequence[int], textures: Sequence[Texture] = [], baseColor: numpy.ndarray = ...) -> OpenGLMesh:
        """
        /// Construct a new owned mesh with lossless compact GPU indices when possible.
          /// Generic constructors retain uint32 buffers for existing raw-VAO clients.
          /// @param vertices Vertex array.
          /// @param indices Triangle index array.
          /// @param textures Optional texture list.
          /// @param baseColor Base color used when textures are absent.
          /// @return The new mesh (requires a current GL context).
        C++: std::shared_ptr<OpenGLMesh> (const std::vector<Vertex> &, const std::vector<unsigned int> &, const std::vector<Texture> &, const glm::vec4 &)
        """

    def indexElementBytes(self) -> int:
        """
        /// GPU element width (2 or 4 bytes); CPU indices remain unsigned int.
          /// Raw VAO users must choose GL_UNSIGNED_SHORT or GL_UNSIGNED_INT accordingly.
          /// @return 2 or 4.
        C++: size_t () const
        """

    def drawFirstIndex(self) -> int:
        """
        /// @return First index of the draw range in the GPU index buffer (non-zero for range views).
        C++: size_t () const
        """

    def drawBaseVertex(self) -> int:
        """
        /// @return Base vertex added to each index when drawing (non-zero for range views).
        C++: int () const
        """

    def setPrimitiveTopology(self, topology: OpenGLMesh.PrimitiveTopology) -> None:
        """
        /**
           * @brief Set the OpenGL primitive topology used by this mesh.
           * @param topology New topology; the index buffer is reinterpreted, not rebuilt.
           */
        C++: void (PrimitiveTopology)
        """

    def primitiveTopology(self) -> OpenGLMesh.PrimitiveTopology:
        """
        /**
           * @return The OpenGL primitive topology used by this mesh.
           */
        C++: PrimitiveTopology () const
        """

    def setLineWidth(self, width: float) -> None:
        """
        /**
           * @brief Set GL line width for PrimitiveTopology::Lines meshes.
           * @param width Width in pixels, clamped to the driver's supported range when drawing.
           */
        C++: void (float)
        """

    def setPointSize(self, size: float) -> None:
        """
        /**
           * @brief Set GL point size for PrimitiveTopology::Points meshes.
           * @param size Point size in pixels.
           */
        C++: void (float)
        """

    def usesPbrShading(self) -> bool:
        """
        /**
           * @return True when this mesh should be rendered by the PBR shader
           *         (Material::usesPbrShading(), cached until invalidateMaterialCache()).
           */
        C++: bool () const
        """

    def materialSortKey(self, mode: OpenGLMesh.RenderMode = OpenGLMesh.RenderMode.All, materialOverride: Material | None = None) -> int:
        """
        /**
           * @brief Compact key for ordering opaque draws by material/texture state.
           * @param mode Render mode the draws will use.
           * @param materialOverride Material to key on instead of #material, or nullptr.
           * @return Hash-like key mixing the PBR flag, texture ids and state flags; sort only, not unique.
           */
        C++: uint64_t (RenderMode, const raisin::Material *) const
        """

    def updateMesh(self, vertices: Sequence[Vertex], indices: Sequence[int]) -> None:
        """
        /**
           * @brief Update mesh data on the GPU.
           *
           * Recomputes tangents. A mesh sharing buffers with others is first given its
           * own private buffers, so other users are unaffected.
           * @param vertices New vertex data.
           * @param indices New index data; an empty vector keeps the current indices.
           */
        C++: void (const std::vector<Vertex> &, const std::vector<unsigned int> &)
        """

    def updateIndices(self, indices: Sequence[int]) -> None:
        """
        /**
           * @brief Replace the triangle index buffer, including clearing it with an empty vector.
           * @param indices New indices into the current vertex buffer.
           */
        C++: void (const std::vector<unsigned int> &)
        """

    def localBoundingSphere(self) -> tuple:
        """
        /// Conservative local sphere, cached until geometry is uploaded again.
          /// Returns false when GPU-only uploads have detached the CPU geometry.
          /// @param[out] center AABB center in object space.
          /// @param[out] radius Half-diagonal of the AABB.
          /// @return False if there are no vertices, a position is non-finite, or the
          ///         GPU data no longer matches #vertices.
        C++: bool (glm::vec3 &, float &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def localBoundingBox(self) -> tuple:
        """
        /// Object-space axis-aligned bounds, same caching and failure cases as localBoundingSphere().
          /// @param[out] center AABB center.
          /// @param[out] halfExtents AABB half extents.
          /// @return False when bounds are unavailable.
        C++: bool (glm::vec3 &, glm::vec3 &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    @overload
    def updateVertices(self, vertices: Sequence[Vertex]) -> None:
        """
        /**
           * @brief Update only vertex data on the GPU, keeping the existing index buffer.
           *
           * Same as `updateVertices(vertices, true)`.
           * @param vertices New vertex data.
           */
        C++: void (const std::vector<Vertex> &)
        """

    @overload
    def updateVertices(self, vertices: Sequence[Vertex], recomputeTangents: bool) -> None:
        """
        /**
           * @brief Update only vertex data on the GPU, optionally preserving caller-provided tangents.
           * @param vertices New vertex data.
           * @param recomputeTangents True to rebuild tangents from the index buffer.
           */
        C++: void (const std::vector<Vertex> &, bool)
        """

    def vertexDisplacementUvStats(self) -> numpy.ndarray:
        """
        /**
           * @brief Mesh-average UV metrics consumed by heightmap vertex displacement:
           * xy = |dP/du|, |dP/dv| in object units per raw UV unit (area weighted),
           * zw = mean raw-UV edge length. Zero when the mesh has no usable UVs.
           * @return The packed statistics.
           */
        C++: glm::vec4 () const
        """

    def updateVertexRange(self, firstVertex: int, vertices: Vertex | None, count: int) -> bool:
        """
        /**
           * @brief Update a contiguous vertex range while preserving the index buffer.
           *
           * Updates both the CPU copy and the GPU buffer; tangents are not recomputed.
           * @param firstVertex Index of the first vertex to replace.
           * @param vertices Replacement vertices.
           * @param count Number of vertices.
           * @return False (nothing changed) if the range is empty or out of bounds.
           */
        C++: bool (size_t, const Vertex *, size_t)
        """

    def uploadVertices(self, vertices: Vertex | None, count: int) -> None:
        """
        /**
           * @brief Upload vertex data without copying it into the CPU-side mesh cache.
           *
           * Unless @p vertices is #vertices itself, CPU-based bounds become unavailable
           * (see localBoundingSphere()).
           * @param vertices Pointer to vertex data.
           * @param count Number of vertices.
           */
        C++: void (const Vertex *, size_t)
        """

    def applyTextureFilteringQuality(self, anisotropy: float, mipLodBias: float) -> int:
        """
        /**
           * @brief Apply sampler quality to all 2D textures owned by this mesh/material.
           *
           * Modifies the GL texture objects, so other meshes using the same textures are affected.
           * @param anisotropy Requested anisotropic filtering level.
           * @param mipLodBias Mip LOD bias.
           * @return Number of unique texture ids updated.
           */
        C++: size_t (float, float) const
        """

    def collectMaterialTextureIds(self) -> tuple:
        """
        /**
           * @brief Append non-environment 2D material texture ids used by this mesh.
           *
           * Environment/IBL maps are renderer resources and are reported separately by
           * RayraiWindow diagnostics.
           * @param[in,out] textureIds Non-zero ids are appended (duplicates are kept).
           */
        C++: void (std::vector<unsigned int> &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def accumulateGeometryDiagnostics(self, diagnostics: OpenGLMesh.GeometryDiagnostics, tinyTriangleAreaThreshold: float) -> None:
        """
        /**
           * @brief Accumulate CPU-side triangle statistics for diagnostics.
           * @param[in,out] diagnostics Totals to add to.
           * @param tinyTriangleAreaThreshold Triangles with smaller (non-zero) area count as tiny.
           */
        C++: void (GeometryDiagnostics &, float) const
        """

    def collectMeshDiagnostics(self, tinyTriangleAreaThreshold: float) -> tuple:
        """
        /**
           * @brief Append per-mesh geometry/material diagnostics for structured JSON output.
           * @param[in,out] diagnostics One entry is appended.
           * @param tinyTriangleAreaThreshold Triangles with smaller (non-zero) area count as tiny.
           */
        C++: void (std::vector<MeshDiagnostics> &, float) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def invalidateMaterialCache(self) -> None:
        """
        /**
           * @brief Refresh cached material-derived flags after direct material edits.
           *
           * Call after changing #material or #textures in place so usesPbrShading()
           * and the texture-slot cache are re-evaluated.
           */
        C++: void () const
        """

class Shader:
    def __init__(self, viewer: RayraiWindow) -> None: ...

    class BinaryCacheStats:
        def __init__(self) -> None: ...

        @property
        def hits(self) -> int: ...

        @hits.setter
        def hits(self, arg: int, /) -> None: ...

        @property
        def misses(self) -> int: ...

        @misses.setter
        def misses(self, arg: int, /) -> None: ...

        @property
        def stores(self) -> int: ...

        @stores.setter
        def stores(self, arg: int, /) -> None: ...

        @property
        def coordinatedWaits(self) -> int: ...

        @coordinatedWaits.setter
        def coordinatedWaits(self, arg: int, /) -> None: ...

    @staticmethod
    def invalidateBindingCache() -> None:
        """
        /// Call after restoring/changing GL_CURRENT_PROGRAM through raw OpenGL.
        C++: void ()
        """

    @property
    def ID(self) -> int: ...

    @ID.setter
    def ID(self, arg: int, /) -> None: ...

    def compile(self, vertexSource: str, fragmentSource: str) -> bool:
        """
        /**
           * @brief Compiles the shader from vertex and fragment source code and links them.
           * @param vertexSource The source code for the vertex shader.
           * @param fragmentSource The source code for the fragment shader.
           *
           * If this Shader already holds a compiled program, the prior program is kept
           * until the new one compiles and links successfully (two-phase replacement),
           * so a failed compile leaves the last known-good program bound rather than a
           * dead/zero program.
           * @return true if a valid program is now active; false if compile/link failed
           *         (in which case the previous program, if any, is preserved).
           */
        C++: bool (const char *, const char *)
        """

    def compileCompute(self, computeSource: str) -> bool:
        """
        /**
           * @brief Compiles a compute shader program.
           * @param computeSource The source code for the compute shader.
           * @return true on success; false if compile/link failed (previous program,
           *         if any, is preserved).
           */
        C++: bool (const char *)
        """

    @staticmethod
    def setCompileThreadCount(threadCount: int) -> None:
        """
        /**
           * @brief Configure the driver-side shader compile thread budget.
           *
           * Passes @p threadCount to `glMaxShaderCompilerThreadsARB`. The default
           * (set by RayraiWindow's constructor) is 1, which keeps shader compilation
           * single-threaded — matching rayrai's overall single-threaded default. A
           * value greater than 1 lets the driver compile vertex/fragment stages and
           * multiple programs concurrently on its own thread pool; useful only when
           * the host application opts into multi-threading.
           *
           * Must be called on the thread that owns the GL context, before the first
           * shader compile; it applies the count to that context at once (other
           * contexts get it before their first compile). Subsequent calls update the
           * driver hint; programs already compiled are unaffected. No-op on drivers
           * without KHR/ARB_parallel_shader_compile.
           */
        C++: void (unsigned int)
        """

    @staticmethod
    def binaryCacheStats() -> Shader.BinaryCacheStats:
        """
        /**
           * @brief Read the current hit/miss/store counters.
           *
           * Useful in tests that need to verify a second run actually picks the
           * cached binary rather than recompiling. Counters are process-global; use
           * `resetBinaryCacheStats` to scope a measurement window.
           */
        C++: BinaryCacheStats ()
        """

    @staticmethod
    def resetBinaryCacheStats() -> None:
        """
        /**
           * @brief Reset the cache hit/miss/store counters to zero.
           */
        C++: void ()
        """

    @staticmethod
    def setBinaryCacheConfig(enabled: bool, directory: str, logOperations: bool) -> None:
        """
        /**
           * @brief Configure the persistent shader-binary cache.
           *
           * Wired through RayraiWindow's constructor. When @p enabled is true and the
           * GL driver advertises at least one program binary format, compiled
           * programs are persisted under @p directory (or an auto-picked path under
           * `$XDG_CACHE_HOME`/`$HOME` if @p directory is empty) keyed by GL
           * vendor/renderer/version + GLSL source. Subsequent runs skip the GLSL
           * compile entirely and load the cached binary.
           *
           * @param enabled         False to bypass the cache entirely.
           * @param directory       Explicit cache directory. Empty for auto.
           * @param logOperations   True to log hits/misses/stores to stderr.
           */
        C++: void (bool, std::string, bool)
        """

    def use(self) -> None:
        """
        /**
           * @brief Activate the shader program for rendering.
           */
        C++: void ()
        """

    @overload
    def setBool(self, name: str, value: bool) -> None:
        """
        /**
           * @brief Set a boolean uniform.
           * @param name Uniform name.
           * @param value Boolean value.
           */
        C++: void (std::string_view, bool) const
        """

    @overload
    def setBool(self, location: int, value: bool) -> None:
        """C++: void (int, bool) const"""

    @overload
    def setInt(self, name: str, value: int) -> None:
        """
        /**
           * @brief Set an integer uniform.
           * @param name Uniform name.
           * @param value Integer value.
           */
        C++: void (std::string_view, int) const
        """

    @overload
    def setInt(self, location: int, value: int) -> None:
        """C++: void (int, int) const"""

    @overload
    def setFloat(self, name: str, value: float) -> None:
        """
        /**
           * @brief Set a float uniform.
           * @param name Uniform name.
           * @param value Float value.
           */
        C++: void (std::string_view, float) const
        """

    @overload
    def setFloat(self, location: int, value: float) -> None:
        """C++: void (int, float) const"""

    @overload
    def setVec2(self, name: str, value: numpy.ndarray) -> None:
        """
        /**
           * @brief Set a vec2 uniform.
           * @param name Uniform name.
           * @param value Vector value.
           */
        C++: void (std::string_view, const glm::vec2 &) const
        """

    @overload
    def setVec2(self, location: int, value: numpy.ndarray) -> None:
        """C++: void (int, const glm::vec2 &) const"""

    @overload
    def setVec2(self, name: str, x: float, y: float) -> None:
        """
        /**
           * @brief Set a vec2 uniform from components.
           * @param name Uniform name.
           * @param x X component.
           * @param y Y component.
           */
        C++: void (std::string_view, float, float) const
        """

    @overload
    def setVec2(self, location: int, x: float, y: float) -> None:
        """C++: void (int, float, float) const"""

    @overload
    def setVec3(self, name: str, value: numpy.ndarray) -> None:
        """
        /**
           * @brief Set a vec3 uniform.
           * @param name Uniform name.
           * @param value Vector value.
           */
        C++: void (std::string_view, const glm::vec3 &) const
        """

    @overload
    def setVec3(self, location: int, value: numpy.ndarray) -> None:
        """C++: void (int, const glm::vec3 &) const"""

    @overload
    def setVec3(self, name: str, x: float, y: float, z: float) -> None:
        """
        /**
           * @brief Set a vec3 uniform from components.
           * @param name Uniform name.
           * @param x X component.
           * @param y Y component.
           * @param z Z component.
           */
        C++: void (std::string_view, float, float, float) const
        """

    @overload
    def setVec3(self, location: int, x: float, y: float, z: float) -> None:
        """C++: void (int, float, float, float) const"""

    @overload
    def setVec4(self, name: str, value: numpy.ndarray) -> None:
        """
        /**
           * @brief Set a vec4 uniform.
           * @param name Uniform name.
           * @param value Vector value.
           */
        C++: void (std::string_view, const glm::vec4 &) const
        """

    @overload
    def setVec4(self, location: int, value: numpy.ndarray) -> None:
        """C++: void (int, const glm::vec4 &) const"""

    @overload
    def setVec4(self, name: str, x: float, y: float, z: float, w: float) -> None:
        """
        /**
           * @brief Set a vec4 uniform from components.
           * @param name Uniform name.
           * @param x X component.
           * @param y Y component.
           * @param z Z component.
           * @param w W component.
           */
        C++: void (std::string_view, float, float, float, float) const
        """

    @overload
    def setVec4(self, location: int, x: float, y: float, z: float, w: float) -> None:
        """C++: void (int, float, float, float, float) const"""

    @overload
    def setIVec4(self, name: str, value: numpy.ndarray) -> None:
        """
        /**
           * @brief Set an ivec4 uniform (not cached).
           * @param name Uniform name.
           * @param value Vector value.
           */
        C++: void (std::string_view, const glm::ivec4 &) const
        """

    @overload
    def setIVec4(self, location: int, value: numpy.ndarray) -> None:
        """C++: void (int, const glm::ivec4 &) const"""

    @overload
    def setMat2(self, name: str, mat: numpy.ndarray) -> None:
        """
        /**
           * @brief Set a mat2 uniform.
           * @param name Uniform name.
           * @param mat Matrix value.
           */
        C++: void (std::string_view, const glm::mat2 &) const
        """

    @overload
    def setMat2(self, location: int, mat: numpy.ndarray) -> None:
        """C++: void (int, const glm::mat2 &) const"""

    @overload
    def setMat3(self, name: str, mat: numpy.ndarray) -> None:
        """
        /**
           * @brief Set a mat3 uniform.
           * @param name Uniform name.
           * @param mat Matrix value.
           */
        C++: void (std::string_view, const glm::mat3 &) const
        """

    @overload
    def setMat3(self, location: int, mat: numpy.ndarray) -> None:
        """C++: void (int, const glm::mat3 &) const"""

    @overload
    def setMat4(self, name: str, mat: numpy.ndarray) -> None:
        """
        /**
           * @brief Set a mat4 uniform.
           * @param name Uniform name.
           * @param mat Matrix value.
           */
        C++: void (std::string_view, const glm::mat4 &) const
        """

    @overload
    def setMat4(self, location: int, mat: numpy.ndarray) -> None:
        """C++: void (int, const glm::mat4 &) const"""

    def cachedUniformLocation(self, name: str) -> int:
        """C++: int (std::string_view) const"""

    def cacheIdentity(self) -> int:
        """
        /// Identity of the *current* linked program, for external uniform-location caches.
          /// Not the GL program name: drivers reuse the lowest free name, so a deleted Shader and
          /// its replacement can share both a heap address and an ID, which would let a stale
          /// location cache survive. This counter is unique for the process instead.
        C++: uint64_t () const
        """

    def uniformGeneration(self) -> int:
        """C++: uint64_t () const"""

class Texture:
    def __init__(self) -> None: ...

    @property
    def id(self) -> int: ...

    @id.setter
    def id(self, arg: int, /) -> None: ...

    @property
    def type(self) -> str: ...

    @type.setter
    def type(self, arg: str, /) -> None: ...

class Vertex:
    def __init__(self) -> None: ...

    @property
    def Position(self) -> numpy.ndarray: ...

    @Position.setter
    def Position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def Normal(self) -> numpy.ndarray: ...

    @Normal.setter
    def Normal(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def TexCoords(self) -> numpy.ndarray: ...

    @TexCoords.setter
    def TexCoords(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def TexCoords2(self) -> numpy.ndarray: ...

    @TexCoords2.setter
    def TexCoords2(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def Tangent(self) -> numpy.ndarray: ...

    @Tangent.setter
    def Tangent(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def Color(self) -> numpy.ndarray: ...

    @Color.setter
    def Color(self, arg: numpy.ndarray, /) -> None: ...

class AdditionalLight:
    def __init__(self) -> None: ...

    @property
    def type(self) -> LightType: ...

    @type.setter
    def type(self, arg: LightType, /) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def direction(self) -> numpy.ndarray: ...

    @direction.setter
    def direction(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def ambient(self) -> numpy.ndarray: ...

    @ambient.setter
    def ambient(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def diffuse(self) -> numpy.ndarray: ...

    @diffuse.setter
    def diffuse(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def specular(self) -> numpy.ndarray: ...

    @specular.setter
    def specular(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def constant(self) -> float: ...

    @constant.setter
    def constant(self, arg: float, /) -> None: ...

    @property
    def linear(self) -> float: ...

    @linear.setter
    def linear(self, arg: float, /) -> None: ...

    @property
    def quadratic(self) -> float: ...

    @quadratic.setter
    def quadratic(self, arg: float, /) -> None: ...

    @property
    def spotInnerCos(self) -> float: ...

    @spotInnerCos.setter
    def spotInnerCos(self, arg: float, /) -> None: ...

    @property
    def spotOuterCos(self) -> float: ...

    @spotOuterCos.setter
    def spotOuterCos(self, arg: float, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def areaSize(self) -> numpy.ndarray: ...

    @areaSize.setter
    def areaSize(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def areaRight(self) -> numpy.ndarray: ...

    @areaRight.setter
    def areaRight(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def areaUp(self) -> numpy.ndarray: ...

    @areaUp.setter
    def areaUp(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def projectorMap(self) -> int: ...

    @projectorMap.setter
    def projectorMap(self, arg: int, /) -> None: ...

    @property
    def projectorStrength(self) -> float: ...

    @projectorStrength.setter
    def projectorStrength(self, arg: float, /) -> None: ...

    @property
    def projectorUvScaleOffset(self) -> numpy.ndarray: ...

    @projectorUvScaleOffset.setter
    def projectorUvScaleOffset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def negative(self) -> bool: ...

    @negative.setter
    def negative(self, arg: bool, /) -> None: ...

    @property
    def temperatureEnabled(self) -> bool: ...

    @temperatureEnabled.setter
    def temperatureEnabled(self, arg: bool, /) -> None: ...

    @property
    def temperatureKelvin(self) -> float: ...

    @temperatureKelvin.setter
    def temperatureKelvin(self, arg: float, /) -> None: ...

    @property
    def distanceFadeEnabled(self) -> bool: ...

    @distanceFadeEnabled.setter
    def distanceFadeEnabled(self, arg: bool, /) -> None: ...

    @property
    def distanceFadeBegin(self) -> float: ...

    @distanceFadeBegin.setter
    def distanceFadeBegin(self, arg: float, /) -> None: ...

    @property
    def distanceFadeShadow(self) -> float: ...

    @distanceFadeShadow.setter
    def distanceFadeShadow(self, arg: float, /) -> None: ...

    @property
    def distanceFadeLength(self) -> float: ...

    @distanceFadeLength.setter
    def distanceFadeLength(self, arg: float, /) -> None: ...

    @property
    def castsShadows(self) -> bool: ...

    @castsShadows.setter
    def castsShadows(self, arg: bool, /) -> None: ...

class BakedIrradianceGrid:
    def __init__(self) -> None: ...

    @property
    def dimensions(self) -> numpy.ndarray: ...

    @dimensions.setter
    def dimensions(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def minimum(self) -> numpy.ndarray: ...

    @minimum.setter
    def minimum(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def maximum(self) -> numpy.ndarray: ...

    @maximum.setter
    def maximum(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def probes(self) -> list[list[numpy.ndarray]]: ...

    @probes.setter
    def probes(self, arg: Sequence[Sequence[numpy.ndarray]], /) -> None: ...

    def validate(self) -> None:
        """
        /// Throws std::invalid_argument for bad dimensions/bounds, a probe count
          /// mismatch, or non-finite / out-of-range coefficients.
        C++: void () const
        """

class Camera:
    @overload
    def __init__(self, viewer: RayraiWindow, position: numpy.ndarray = ..., up: numpy.ndarray = ..., yaw: float = -135.0, pitch: float = -30.0) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, rgbCam: raisimpy.RGBCamera) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, rgbCam: raisimpy.RGBCamera) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, depthCam: raisimpy.DepthCamera) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, depthCam: raisimpy.DepthCamera) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, properties: raisimpy.DepthCameraProperties) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, properties: raisimpy.RGBCameraProperties) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, segmentationCam: raisimpy.SegmentationCamera) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, segmentationCam: raisimpy.SegmentationCamera) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, properties: raisimpy.SegmentationCameraProperties) -> None: ...

    class ProjectionMode(enum.Enum):
        PERSPECTIVE = 0

        ORTHOGRAPHIC = 1

    class RenderTargetDiagnostics:
        def __init__(self) -> None: ...

        @property
        def name(self) -> str: ...

        @name.setter
        def name(self, arg: str, /) -> None: ...

        @property
        def framebuffer(self) -> int: ...

        @framebuffer.setter
        def framebuffer(self, arg: int, /) -> None: ...

        @property
        def colorTexture(self) -> int: ...

        @colorTexture.setter
        def colorTexture(self, arg: int, /) -> None: ...

        @property
        def depthTexture(self) -> int: ...

        @depthTexture.setter
        def depthTexture(self, arg: int, /) -> None: ...

        @property
        def colorRenderbuffer(self) -> int: ...

        @colorRenderbuffer.setter
        def colorRenderbuffer(self, arg: int, /) -> None: ...

        @property
        def depthRenderbuffer(self) -> int: ...

        @depthRenderbuffer.setter
        def depthRenderbuffer(self, arg: int, /) -> None: ...

        @property
        def width(self) -> int: ...

        @width.setter
        def width(self, arg: int, /) -> None: ...

        @property
        def height(self) -> int: ...

        @height.setter
        def height(self, arg: int, /) -> None: ...

        @property
        def samples(self) -> int: ...

        @samples.setter
        def samples(self, arg: int, /) -> None: ...

        @property
        def colorFormat(self) -> str: ...

        @colorFormat.setter
        def colorFormat(self, arg: str, /) -> None: ...

        @property
        def depthFormat(self) -> str: ...

        @depthFormat.setter
        def depthFormat(self, arg: str, /) -> None: ...

        @property
        def estimatedBytes(self) -> int: ...

        @estimatedBytes.setter
        def estimatedBytes(self, arg: int, /) -> None: ...

        @property
        def ready(self) -> bool: ...

        @ready.setter
        def ready(self, arg: bool, /) -> None: ...

        @property
        def framebufferStatus(self) -> int: ...

        @framebufferStatus.setter
        def framebufferStatus(self, arg: int, /) -> None: ...

        @property
        def framebufferStatusName(self) -> str: ...

        @framebufferStatusName.setter
        def framebufferStatusName(self, arg: str, /) -> None: ...

        @property
        def framebufferComplete(self) -> bool: ...

        @framebufferComplete.setter
        def framebufferComplete(self, arg: bool, /) -> None: ...

    class SensorStorageMode(enum.Enum):
        SENSOR_SETTER = 0

        CUSTOM_BUFFER = 1

    def instanceId(self) -> int:
        """
        /// @return Process-unique id assigned at construction (never reused).
        C++: std::uint64_t () const
        """

    def update(self, processKeyboard: bool = True) -> None:
        """
        /**
           * @brief Update camera orientation and handle input.
           *
           * Recomputes #front, #right and #up from #yaw / #pitch. When a target
           * object or fixed distance is active, the camera is placed on the orbit
           * around getTargetPosition() instead.
           * @param processKeyboard If true, WASD and Space (along #worldUp) move the
           *        camera by #movementSpeed per second assuming a fixed 1/60 s step.
           *        Ignored while orbiting a target or with a fixed distance.
           */
        C++: void (bool)
        """

    def getViewMatrix(self) -> numpy.ndarray:
        """
        /**
           * @brief Get the view matrix for the current camera pose.
           * @return View matrix in column-major GLM format.
           */
        C++: glm::mat4 () const
        """

    def getProjectionMatrix(self) -> numpy.ndarray:
        """
        /**
           * @brief Get the projection matrix for the current camera state.
           * @return Projection matrix in column-major GLM format.
           */
        C++: glm::mat4 () const
        """

    def getProjectionClipRange(self) -> numpy.ndarray:
        """
        /**
           * @brief Near/far distances actually used by getProjectionMatrix().
           *
           * Near is the smaller of #nearPlane and #zNear (when positive) and far the
           * larger of #farPlane and #zFar, with fallbacks for invalid values.
           * @return (near, far) in metres.
           */
        C++: glm::vec2 () const
        """

    def setProjectionJitterPixels(self, jitterPixels: numpy.ndarray) -> None:
        """
        /**
           * @brief Set subpixel projection jitter in pixels for temporal AA style viewer rendering.
           * @param jitterPixels Offset in render-target pixels.
           */
        C++: void (const glm::vec2 &)
        """

    def clearProjectionJitter(self) -> None:
        """
        /**
           * @brief Clear projection jitter.
           */
        C++: void ()
        """

    def projectionJitterPixels(self) -> numpy.ndarray:
        """
        /// @return Current projection offset in render-target pixels (zero for sensor opt-out).
        C++: glm::vec2 () const
        """

    def getPosition(self) -> numpy.ndarray:
        """
        /**
           * @brief Get the current camera position.
           * @return Position in world coordinates.
           */
        C++: glm::vec3 () const
        """

    def mouseDelta(self, yawOffset: float, pitchOffset: float) -> None:
        """
        /**
           * @brief Apply mouse delta to yaw/pitch or pan in orthographic mode.
           *
           * Perspective: `yaw -= yawOffset`, `pitch += pitchOffset` (pitch clamped to
           * +-89 deg). Orthographic: pans the position by 0.05 m per unit. Call update()
           * afterwards to refresh the direction vectors.
           * @param yawOffset Delta yaw in degrees (screen-space delta).
           * @param pitchOffset Delta pitch in degrees (screen-space delta).
           */
        C++: void (float, float)
        """

    def moveForward(self, steps: float) -> None:
        """
        /**
           * @brief Move forward/backward or zoom in orthographic mode.
           *
           * Free perspective camera: moves `steps` metres along #front. Orthographic:
           * scales #orthoScale by 0.9^steps (min 0.1), so positive zooms in. Orbiting
           * a target or with a fixed distance: scales the orbit distance by 1.2^steps
           * (min 0.05 m), so positive steps move away from the target.
           * @param steps Signed step count (e.g. mouse-wheel delta).
           */
        C++: void (float)
        """

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def front(self) -> numpy.ndarray: ...

    @front.setter
    def front(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def target(self) -> numpy.ndarray: ...

    @target.setter
    def target(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def up(self) -> numpy.ndarray: ...

    @up.setter
    def up(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def right(self) -> numpy.ndarray: ...

    @right.setter
    def right(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def worldUp(self) -> numpy.ndarray: ...

    @worldUp.setter
    def worldUp(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def yaw(self) -> float: ...

    @yaw.setter
    def yaw(self, arg: float, /) -> None: ...

    @property
    def pitch(self) -> float: ...

    @pitch.setter
    def pitch(self, arg: float, /) -> None: ...

    @property
    def movementSpeed(self) -> float: ...

    @movementSpeed.setter
    def movementSpeed(self, arg: float, /) -> None: ...

    @property
    def zoom(self) -> float: ...

    @zoom.setter
    def zoom(self, arg: float, /) -> None: ...

    @property
    def aspect(self) -> float: ...

    @aspect.setter
    def aspect(self, arg: float, /) -> None: ...

    @property
    def horizontalFovRad(self) -> float: ...

    @horizontalFovRad.setter
    def horizontalFovRad(self, arg: float, /) -> None: ...

    @property
    def nearPlane(self) -> float: ...

    @nearPlane.setter
    def nearPlane(self, arg: float, /) -> None: ...

    @property
    def farPlane(self) -> float: ...

    @farPlane.setter
    def farPlane(self, arg: float, /) -> None: ...

    @property
    def orthoScale(self) -> float: ...

    @orthoScale.setter
    def orthoScale(self, arg: float, /) -> None: ...

    @property
    def zNear(self) -> float: ...

    @zNear.setter
    def zNear(self, arg: float, /) -> None: ...

    @property
    def zFar(self) -> float: ...

    @zFar.setter
    def zFar(self, arg: float, /) -> None: ...

    def setCameraFixedDistance(self, fixed: bool) -> None:
        """
        /**
           * @brief Lock camera distance while orbiting.
           * @param fixed True to keep distance constant.
           */
        C++: void (bool)
        """

    def isCameraFixedDistance(self) -> bool:
        """
        /** @return True when the orbit distance is locked. */
        C++: bool () const
        """

    def cameraFixedDistance(self) -> float:
        """
        /** @return Current orbit distance in metres (tracked while orbiting, locked when fixed). */
        C++: float () const
        """

    def setCameraFixedTarget(self, fixed: bool) -> None:
        """
        /**
           * @brief Lock the target position to a fixed point.
           * @param fixed True to use the #target field as the orbit/look-at point.
           */
        C++: void (bool)
        """

    def isCameraFixedTarget(self) -> bool:
        """
        /** @return True when the camera target is locked to a fixed point. */
        C++: bool () const
        """

    def setTargetObject(self, target: raisimpy.Object | None) -> None:
        """
        /**
           * @brief Set a raisim object to track as the camera target.
           *
           * A non-null object clears the fixed-target lock. The camera never reads the
           * object: the RayraiWindow that renders with this camera follows its origin
           * in the rendered physics frame (followTargetObject()), and releases the
           * camera once the object is gone. A fixed distance locked before the first
           * followed frame is measured on that frame.
           * @param target Raisim object to follow, or nullptr to clear.
           */
        C++: void (raisim::Object *)
        """

    def releaseTarget(self) -> None:
        """
        /// Stop following: clears the target object and the fixed-target and fixed-distance locks.
        C++: void ()
        """

    def followTargetObject(self, instance: int, position: numpy.ndarray) -> bool:
        """
        /**
           * @brief Internal: the followed #targetObject as rendered, set by RayraiWindow whenever it
           * renders; applications do not call this.
           *
           * A camera that update() placed moves with the object.
           * @param instance RaisimObject::instanceId() of the object's drawable.
           * @param position World position of the object's origin.
           * @return False when another object now has the target's address: the caller releases it.
           */
        C++: bool (uint64_t, const glm::vec3 &)
        """

    def setProjectionMode(self, mode: Camera.ProjectionMode) -> None:
        """
        /**
           * @brief Set projection mode.
           * @param mode Projection mode (perspective or orthographic).
           */
        C++: void (ProjectionMode)
        """

    def getProjectionMode(self) -> Camera.ProjectionMode:
        """
        /**
           * @brief Get the active projection mode.
           * @return Current projection mode.
           */
        C++: ProjectionMode () const
        """

    def setLensModel(self, lens: raisimpy.CameraLensModel, hFovRad: float) -> None:
        """
        /**
           * @brief Configure the camera lens model used for sensor rendering.
           *
           * Only stores the lens and #horizontalFovRad; it does not change #zoom.
           * @param lens Lens model (pinhole or fisheye).
           * @param hFovRad Horizontal field of view in radians.
           */
        C++: void (const raisim::CameraLensModel &, double)
        """

    def getLensModel(self) -> raisimpy.CameraLensModel:
        """
        /**
           * @brief Get the active sensor lens model.
           * @return Reference to the stored lens model.
           */
        C++: const raisim::CameraLensModel &() const
        """

    def isFisheyeLens(self) -> bool:
        """
        /**
           * @return True when the camera is configured for OpenCV/ROS equidistant fisheye output.
           */
        C++: bool () const
        """

    def applyFisheyeRemap(self, fisheyeShader: Shader) -> None:
        """
        /**
           * @brief Resample scene color through the fisheye lens model.
           *
           * No-op unless isFisheyeLens() and scene targets exist. Renders into the
           * post target and blits the result back into the scene color texture.
           * @param fisheyeShader Fisheye remap shader (bound here).
           */
        C++: void (Shader &)
        """

    def getTargetPosition(self) -> numpy.ndarray:
        """
        /**
           * @brief Get the current target position for orbiting.
           * @return #target when the fixed-target lock is on, else the target
           *         object's position, else the world origin.
           */
        C++: glm::vec3 () const
        """

    @property
    def targetObject(self) -> raisimpy.Object | None: ...

    @targetObject.setter
    def targetObject(self, arg: raisimpy.Object, /) -> None: ...

    def ensureRenderTargets(self, width: int, height: int) -> None:
        """
        /**
           * @brief Ensure render targets match the given size.
           *
           * (Re)allocates the scene color/depth targets (and the MSAA target when
           * enabled) and updates #aspect. Requires a current GL context; ignored for
           * non-positive sizes.
           * @param width Render target width.
           * @param height Render target height.
           */
        C++: void (int, int)
        """

    def resizeRenderTargets(self, width: int, height: int) -> None:
        """
        /**
           * @brief Resize render targets, retaining depth-only allocation when applicable.
           * @param width Render target width.
           * @param height Render target height.
           */
        C++: void (int, int)
        """

    @overload
    def setSceneMsaaSamples(self, samples: int) -> None:
        """
        /**
           * @brief Set scene-color MSAA samples for future scene FBO rendering.
           *
           * Values <= 1 disable MSAA. The resolved color/depth are still written into the existing
           * single-sample texture targets so postprocessing and readback APIs keep working.
           * @param samples Requested sample count; values above 1 are clamped to [2, GL_MAX_SAMPLES].
           */
        C++: void (int)
        """

    @overload
    def setSceneMsaaSamples(self, samples: int, keepAllocation: bool) -> None:
        """
        /**
           * @brief Same as setSceneMsaaSamples(), optionally keeping an existing multisampled target.
           *
           * With keepAllocation, disabling MSAA keeps the multisampled renderbuffers so re-enabling the
           * same sample count reuses them instead of reallocating. Every scene render clears the target
           * before drawing, so reuse cannot change the image. A later call without keepAllocation that
           * leaves MSAA disabled releases the kept target.
           * @param samples Requested sample count, as in the single-argument overload.
           * @param keepAllocation Keep the multisampled renderbuffers when disabling MSAA.
           */
        C++: void (int, bool)
        """

    def sceneMsaaSamples(self) -> int:
        """
        /**
           * @return Current scene-color MSAA sample count. 1 means disabled.
           */
        C++: int () const
        """

    def resolveSceneFboIfNeeded(self, restoreFramebuffer: bool = True) -> None:
        """
        /**
           * @brief Resolve the multisampled scene target into the regular color/depth textures.
           * @param restoreFramebuffer Restore the previous read/draw framebuffer bindings afterwards.
           */
        C++: void (bool)
        """

    def ensurePostRenderTarget(self) -> None:
        """
        /**
           * @brief Ensure the post-process render target exists.
           */
        C++: void ()
        """

    def ensureLinearDepthRenderTarget(self) -> None:
        """
        /**
           * @brief Ensure the linear-depth render target exists.
           */
        C++: void ()
        """

    def bindSceneFbo(self) -> None:
        """
        /**
           * @brief Bind the scene framebuffer for rendering.
           *
           * Binds the MSAA framebuffer when MSAA is enabled, allocating targets on demand.
           */
        C++: void ()
        """

    def bindPostFbo(self) -> None:
        """
        /**
           * @brief Bind the post-process framebuffer for rendering.
           */
        C++: void ()
        """

    def bindLinearDepthFbo(self) -> None:
        """
        /**
           * @brief Bind the linear-depth framebuffer for rendering.
           */
        C++: void ()
        """

    def ensureSegmentationRenderTarget(self) -> None:
        """
        /**
           * @brief Ensure the segmentation render target (RGBA32I color + depth24) exists.
           */
        C++: void ()
        """

    @staticmethod
    def unbindFbo() -> None:
        """
        /**
           * @brief Unbind any framebuffer (bind default).
           */
        C++: void ()
        """

    def rtWidth(self) -> int:
        """
        /**
           * @brief Current render target width.
           * @return Width in pixels.
           */
        C++: int () const
        """

    def rtHeight(self) -> int:
        """
        /**
           * @brief Current render target height.
           * @return Height in pixels.
           */
        C++: int () const
        """

    def beginSceneFboAndClear(self, preserveState: bool = True) -> None:
        """
        /**
           * @brief Bind scene FBO and clear color, depth and stencil with the background color.
           * @param preserveState Restore the color/depth/stencil write masks and scissor
           *        state afterwards; false leaves scissor disabled and masks enabled.
           */
        C++: void (bool)
        """

    def postProcessDoF(self, postShader: Shader, fxaaEnabled: bool = False, bloomEnabled: bool = False, bloomThreshold: float = 0.8199999928474426, bloomStrength: float = 0.18000000715255737, bloomRadius: float = 4.0, bloomKnee: float = 0.2199999988079071, bloomQuality: int = 1, bloomSourceClamp: float = 8.0, bloomAnamorphic: float = 0.0, bloomDirtStrength: float = 0.0, bloomDirtScale: float = 1.0, bloomDirtTexture: int = 0, bloomBlendMode: int = 0, bloomMix: float = 0.05000000074505806, viewerWhiteBalance: numpy.ndarray = ..., viewerSaturation: float = 1.0, screenSpaceAoEnabled: bool = False, screenSpaceAoRadius: float = 2.0, screenSpaceAoStrength: float = 0.0, screenSpaceAoBias: float = 0.019999999552965164, screenSpaceAoSamples: int = 12, screenSpaceAoFalloff: float = 0.07999999821186066, screenSpaceAoRadiusWorldSpace: bool = False, screenSpaceAoMaxPixelRadius: float = 48.0, screenSpaceAoGeometryAware: bool = False, contactAoRadius: float = 0.0, contactAoStrength: float = 0.0, contactAoSamples: int = 8, contactAoFalloff: float = 0.05000000074505806, screenSpaceAoDenoiseEnabled: bool = False, screenSpaceAoDenoiseStrength: float = 0.0, screenSpaceAoDenoiseRadius: int = 1, screenSpaceAoDenoiseDepthSigma: float = 0.03500000014901161, screenSpaceIndirectLightingEnabled: bool = False, screenSpaceIndirectLightingRadius: float = 3.0, screenSpaceIndirectLightingStrength: float = 0.0, screenSpaceIndirectLightingSamples: int = 12, screenSpaceIndirectLightingFalloff: float = 0.2199999988079071, screenSpaceIndirectLightingNormalRejection: float = 0.75, screenSpaceIndirectLightingSaturation: float = 1.0, volumetricFogEnabled: bool = False, volumetricFogDensity: float = 0.0, volumetricFogNoiseScale: float = 6.0, volumetricFogNoiseStrength: float = 0.550000011920929, volumetricFogColor: numpy.ndarray = ..., volumetricFogAnisotropy: float = 0.0, volumetricFogAnimationTimeSeconds: float = 0.0, volumetricFogWindDirection: numpy.ndarray = ..., volumetricFogWindSpeed: float = 0.0, volumetricFogTurbulenceSpeed: float = 0.0, volumetricLightingEnabled: bool = False, volumetricLightStrength: float = 0.0, volumetricLightDecay: float = 0.8799999952316284, volumetricLightSamples: int = 16, volumetricLightScreenPosition: numpy.ndarray = ..., volumetricLightColor: numpy.ndarray = ..., localFogVolumeCount: int = 0, localFogVolumeCenterRadius: Sequence[numpy.ndarray] = ..., localFogVolumeColorDensity: Sequence[numpy.ndarray] = ..., localFogVolumeParams: Sequence[numpy.ndarray] = ..., projectedDecalCount: int = 0, projectedDecalWorldToLocal: Sequence[numpy.ndarray] = ..., projectedDecalColor: Sequence[numpy.ndarray] = ..., projectedDecalParams: Sequence[numpy.ndarray] = ..., projectedDecalAlbedoMaps: Sequence[int] = [0, 0, 0, 0, 0, 0, 0, 0], projectedDecalEmissionMaps: Sequence[int] = [0, 0, 0, 0, 0, 0, 0, 0], projectedDecalNormalMaps: Sequence[int] = [0, 0, 0, 0, 0, 0, 0, 0], projectedDecalOrmMaps: Sequence[int] = [0, 0, 0, 0, 0, 0, 0, 0], projectedDecalTextureParams: Sequence[numpy.ndarray] = ..., projectedDecalSurfaceParams: Sequence[numpy.ndarray] = ..., projectedDecalUvScaleOffset: Sequence[numpy.ndarray] = ..., temporalAaEnabled: bool = False, temporalAaBlend: float = 0.07999999821186066, temporalAaReset: bool = False, postProcessDebugMode: int = 0, aerialPerspectiveEnabled: bool = False, apVolumeTex: int = 0, apStrength: float = 1.0, viewerColorGradePreset: int = 0, viewerColorGradeStrength: float = 1.0, viewerVignetteStrength: float = 0.0, viewerChromaticAberrationStrength: float = 0.0, viewerFilmGrainStrength: float = 0.0, viewerLensCharacterTime: float = 0.0, lightShaftsEnabled: bool = False, lightShaftsStrength: float = 1.0, lightShaftsDecay: float = 0.9649999737739563, lightShaftsDensity: float = 1.0, lightShaftsSunScreenPos: numpy.ndarray = ..., lightShaftsTint: numpy.ndarray = ..., lensFlareGhostStrength: float = 0.0, lensFlareStreakStrength: float = 0.0, lensFlareTint: numpy.ndarray = ..., contactShadowsEnabled: bool = False, contactShadowsLength: float = 0.11999999731779099, contactShadowsStrength: float = 0.699999988079071, contactShadowsThickness: float = 0.07999999821186066, contactShadowsSunScreenPos: numpy.ndarray = ..., contactShadowsLightDirectionWorld: numpy.ndarray = ..., motionBlurEnabled: bool = False, motionBlurDirection: numpy.ndarray = ..., motionBlurStrength: float = 0.0, ssrEnabled: bool = False, ssrStrength: float = 0.0, ssrSteps: int = 24, ssrMaxDistance: float = 6.0, ssrThickness: float = 0.3499999940395355, heatHazeEnabled: bool = False, heatHazeStrength: float = 0.0, heatHazeFrequency: float = 4.0, heatHazeSpeed: float = 1.0, heatHazeMaxY: float = 0.550000011920929, zoomBlurEnabled: bool = False, zoomBlurStrength: float = 0.0, zoomBlurCenter: numpy.ndarray = ..., zoomBlurInnerRadius: float = 0.15000000596046448, posterizeEnabled: bool = False, posterizeLevels: int = 4, starburstEnabled: bool = False, starburstStrength: float = 0.0, starburstSpikes: int = 6, starburstAngleOffset: float = 0.0, starburstRadius: float = 0.3499999940395355, starburstTint: numpy.ndarray = ..., lensDistortionEnabled: bool = False, lensDistortionStrength: float = 0.0, lensDistortionCenter: numpy.ndarray = ..., underwaterEnabled: bool = False, underwaterTint: numpy.ndarray = ..., underwaterCausticStrength: float = 0.5, underwaterCausticScale: float = 1.0, underwaterDepthExtinction: float = 1.0, pixelateEnabled: bool = False, pixelateBlockSize: int = 4, nightVisionEnabled: bool = False, nightVisionStrength: float = 0.0, nightVisionGain: float = 2.5, ditherEnabled: bool = False, ditherStrength: float = 0.004000000189989805, thermalEnabled: bool = False, thermalStrength: float = 0.0, thermalGain: float = 1.5, scanlineEnabled: bool = False, scanlineStrength: float = 0.30000001192092896, scanlineFrequency: float = 360.0, letterboxEnabled: bool = False, letterboxAspect: float = 2.3499999046325684, letterboxColor: numpy.ndarray = ..., aoPrepassShader: Shader | None = None, contactShadowShader: Shader | None = None) -> None:
        """
        /**
           * @brief Apply viewer post-processing.
           *
           * Internal: called by RayraiWindow, which fills the long parameter list from
           * its render-quality and weather settings. The remaining parameters map
           * one-to-one onto post-shader uniforms. When no effect is enabled the pass
           * is skipped and getFinalTexture() keeps returning the scene color.
           * @param postShader Shader configured for DoF, FXAA, and bloom.
           * @param fxaaEnabled True to apply FXAA even when DoF is disabled.
           * @param aoPrepassShader Optional leaner post program that runs the AO prepass, which caches
           *        AO for the denoiser, in place of @p postShader (null: @p postShader runs it).
           * @param contactShadowShader Post program that renders only the unfiltered contact shadows of
           *        the filter taps, into a texture @p postShader then filters (uContactShadowTexture).
           *        Each tap is marched once although up to four pixels share it. Required when
           *        @p postShader declares that sampler and contact shadows are enabled; ignored
           *        otherwise. That program's decal map pool is one slot smaller.
           */
        C++: void (Shader &, bool, bool, float, float, float, float, int, float, float, float, float, unsigned int, int, float, const glm::vec3 &, float, bool, float, float, float, int, float, bool, float, bool, float, float, int, float, bool, float, int, float, bool, float, float, int, float, float, float, bool, float, float, float, const glm::vec3 &, float, float, const glm::vec2 &, float, float, bool, float, float, int, const glm::vec2 &, const glm::vec3 &, int, const std::array<glm::vec4, 8> &, const std::array<glm::vec4, 8> &, const std::array<glm::vec4, 8> &, int, const std::array<glm::mat4, 8> &, const std::array<glm::vec4, 8> &, const std::array<glm::vec4, 8> &, const std::array<unsigned int, 8> &, const std::array<unsigned int, 8> &, const std::array<unsigned int, 8> &, const std::array<unsigned int, 8> &, const std::array<glm::vec4, 8> &, const std::array<glm::vec4, 8> &, const std::array<glm::vec4, 8> &, bool, float, bool, int, bool, unsigned int, float, int, float, float, float, float, float, bool, float, float, float, const glm::vec2 &, const glm::vec3 &, float, float, const glm::vec3 &, bool, float, float, float, const glm::vec2 &, const glm::vec3 &, bool, const glm::vec2 &, float, bool, float, int, float, float, bool, float, float, float, float, bool, float, const glm::vec2 &, float, bool, int, bool, float, int, float, float, const glm::vec3 &, bool, float, const glm::vec2 &, bool, const glm::vec3 &, float, float, float, bool, int, bool, float, float, bool, float, bool, float, float, bool, float, float, bool, float, const glm::vec3 &, Shader *, Shader *)
        """

    def resetTemporalHistory(self) -> None:
        """
        /**
           * @brief Reset the temporal postprocess history before the next temporal frame.
           */
        C++: void ()
        """

    def updateAutoExposure(self, targetKey: float, speed: float, minFactor: float, maxFactor: float) -> None:
        """
        /**
           * @brief Sample mean scene-color luma and lerp the auto-exposure factor.
           *
           * Every 4th call issues an asynchronous PBO readback of the smallest mip of
           * the post-process (or scene) color; a completed earlier readback, if any,
           * yields the mean luma, and `autoExposureFactor_` is smoothed toward
           * `targetKey / mean`. Never blocks: without a ready sample the factor is
           * unchanged. Should be called once per frame, after post-process completes.
           * The stored factor is then applied to the next frame's `pbrExposure`.
           * @param targetKey Desired mean luma (approximately linear, 0..1).
           * @param speed Lerp factor per update, clamped to [0, 1].
           * @param minFactor Lower clamp of the exposure factor.
           * @param maxFactor Upper clamp of the exposure factor.
           */
        C++: void (float, float, float, float)
        """

    def autoExposureFactor(self) -> float:
        """
        /**
           * @return Smoothed auto-exposure multiplier from the previous frame.
           */
        C++: float () const
        """

    def debugAutoExposureIssuedReadbacks(self) -> int:
        """
        /// Internal: @return number of auto-exposure readbacks issued (diagnostics).
        C++: size_t () const
        """

    def debugAutoExposureCompletedReadbacks(self) -> int:
        """
        /// Internal: @return number of auto-exposure readbacks consumed (diagnostics).
        C++: size_t () const
        """

    def debugAutoExposureSkippedIssueFrames(self) -> int:
        """
        /// Internal: @return readback slots skipped because the ring was full or fencing failed (diagnostics).
        C++: size_t () const
        """

    def resetAutoExposure(self) -> None:
        """
        /**
           * @brief Reset auto-exposure to identity (no compensation).
           */
        C++: void ()
        """

    def temporalHistoryValid(self) -> bool:
        """
        /**
           * @return True if a temporal history texture is available for the current render target.
           */
        C++: bool () const
        """

    def temporalHistoryFrameCount(self) -> int:
        """
        /**
           * @return Number of post-process frames copied into temporal history since the last reset.
           */
        C++: int () const
        """

    def ensurePickFramebuffer(self) -> None:
        """
        /**
           * @brief Ensure the picking framebuffer is created.
           */
        C++: void ()
        """

    def readPickIdAt(self, x: int, y: int) -> int:
        """
        /**
           * @brief Read the encoded pick id at a pixel coordinate.
           *
           * Synchronous glReadPixels from the pick framebuffer.
           * @param x X coordinate in render-target pixels from the left.
           * @param y Y coordinate in render-target pixels from the top (flipped internally).
           * @return 24-bit id encoded in the pick texture's RGB bytes; 0 when out of bounds.
           */
        C++: uint32_t (int, int) const
        """

    def getPickFbo(self) -> int:
        """
        /**
           * @brief Get the picking FBO id.
           * @return OpenGL framebuffer id.
           */
        C++: unsigned int () const
        """

    def getFinalTexture(self) -> int:
        """
        /**
           * @brief Get the final texture to display (post or debug).
           * @return Pick texture when setDebugShowPick(true), else the post-process
           *         color if post-processing ran since the last scene bind, else the
           *         scene color texture.
           */
        C++: unsigned int () const
        """

    def getSceneColorTex(self) -> int:
        """
        /**
           * @brief Get the scene color texture id.
           * @return Color texture id, or zero before scene targets are allocated.
           */
        C++: unsigned int () const
        """

    def getSceneDepthTex(self) -> int:
        """
        /**
           * @brief Get the scene depth texture id.
           * @return Depth24/stencil8 texture id, or zero before scene targets are allocated.
           */
        C++: unsigned int () const
        """

    def getSceneFbo(self) -> int:
        """
        /**
           * @brief Get the scene framebuffer id.
           * @return Single-sample scene FBO id; zero before a color render or explicit
           *         ensureRenderTargets().
           */
        C++: unsigned int () const
        """

    def getLinearDepthTexture(self) -> int:
        """
        /**
           * @brief Get the linear depth texture id (R32F).
           * @return Linear depth texture id, or zero before the linear-depth target is allocated.
           */
        C++: unsigned int () const
        """

    def getSegmentationTexture(self) -> int:
        """
        /**
           * @brief Get the segmentation texture id (RGBA32I: world index, body index, label, 0).
           * @return Segmentation texture id, or zero before the segmentation target is allocated.
           */
        C++: unsigned int () const
        """

    def renderTargetDiagnostics(self) -> list[Camera.RenderTargetDiagnostics]:
        """
        /**
           * @brief Return a diagnostics-only snapshot of camera-owned render targets.
           *
           * Queries framebuffer status, so a current GL context is required.
           * @return One entry per target (scene, MSAA, post, temporal, bloom, picking, linear depth, ...).
           */
        C++: std::vector<RenderTargetDiagnostics> () const
        """

    def setAsyncReadbackEnabled(self, enabled: bool, ringSize: int = 3) -> None:
        """
        /**
           * @brief Configure optional PBO-backed readback helpers.
           *
           * Sensor getRawImage() remains synchronous by default. After the configured PBO ring fills, the
           * async helpers return an earlier completed frame and normally lag by `ringSize` calls.
           * @param enabled Enable the async helpers; disabling releases the PBOs.
           * @param ringSize Number of PBOs per ring (min 2).
           */
        C++: void (bool, size_t)
        """

    def isAsyncReadbackEnabled(self) -> bool:
        """
        /// @return True when the async readback helpers are enabled.
        C++: bool () const
        """

    def setBackgroundColor(self, c: numpy.ndarray) -> None:
        """
        /**
           * @brief Set the background clear color using legacy 0..255 RGBA channels.
           * @param c RGBA color where each channel is in [0, 255].
           *
           * Prefer setBackgroundColorRgb255() in new code to make the range explicit.
           */
        C++: void (const glm::vec4 &)
        """

    def setBackgroundColorRgb255(self, c: numpy.ndarray) -> None:
        """
        /**
           * @brief Set the background clear color using 0..255 RGBA channels.
           * @param c RGBA color where each channel is in [0, 255].
           */
        C++: void (const glm::vec4 &)
        """

    def setBackgroundColorLinear(self, c: numpy.ndarray) -> None:
        """
        /**
           * @brief Set the background clear color using linear 0..1 RGBA channels.
           * @param c Linear RGBA color where each channel is in [0, 1].
           */
        C++: void (const glm::vec4 &)
        """

    def getBackgroundColor(self) -> numpy.ndarray:
        """
        /**
           * @brief Get the background clear color.
           * @return Legacy RGBA color in [0, 255].
           */
        C++: glm::vec4 () const
        """

    def getBackgroundColorLinear(self) -> numpy.ndarray:
        """
        /**
           * @brief Get the background clear color in linear 0..1 channels.
           * @return Linear RGBA color in [0, 1].
           */
        C++: glm::vec4 () const
        """

    def setGamma(self, g: float) -> None:
        """
        /**
           * @brief Set the display gamma (output encoded as `c^(1/g)`; default 1).
           *
           * Applied to the background clear color and passed to the scene shaders.
           * @param g Gamma value.
           */
        C++: void (float)
        """

    def getGamma(self) -> float:
        """
        /**
           * @brief Get gamma correction value.
           * @return Gamma value.
           */
        C++: float () const
        """

    def setDOFEnabled(self, e: bool) -> None:
        """
        /**
           * @brief Enable or disable DoF.
           * @param e True to enable.
           */
        C++: void (bool)
        """

    def getDOFEnabled(self) -> bool:
        """
        /**
           * @brief Get DoF enabled state.
           * @return True if DoF is enabled.
           */
        C++: bool () const
        """

    def getDOFFocusDistance(self) -> float:
        """
        /// @return DoF focus distance in metres.
        C++: float () const
        """

    def getDOFFocusRange(self) -> float:
        """
        /// @return Width of the fully sharp band in metres.
        C++: float () const
        """

    def getDOFMaxRadius(self) -> float:
        """
        /// @return Maximum blur (bokeh) radius in pixels.
        C++: float () const
        """

    def setDOFParams(self, focusDist: float, focusRange: float, maxRadius: float) -> None:
        """
        /**
           * @brief Set DoF parameters.
           * @param focusDist Focus distance in metres.
           * @param focusRange Width of the fully sharp band centred on the focus distance, in metres.
           * @param maxRadius Max blur (bokeh) radius in pixels.
           */
        C++: void (float, float, float)
        """

    def setDebugShowPick(self, e: bool) -> None:
        """
        /**
           * @brief Toggle rendering of the pick buffer for debugging.
           * @param e True to show pick buffer.
           */
        C++: void (bool)
        """

    def isDebugShowingPick(self) -> bool:
        """
        /**
           * @brief Check if the pick buffer is shown for debugging.
           * @return True if pick buffer is displayed.
           */
        C++: bool () const
        """

    def readRgba(self, flipVertical: bool = True) -> numpy.ndarray: ...

    def readSceneColorBgra(self, flipVertical: bool = True) -> numpy.ndarray: ...

    def readLinearDepth(self, flipVertical: bool = True) -> numpy.ndarray: ...

    def readSceneColorRgbaAsync(self, flipVertical: bool = True) -> numpy.ndarray | None:
        """Return uint8 RGBA or None while the readback ring is pending."""

    def readLinearDepthAsync(self, flipVertical: bool = True) -> numpy.ndarray | None:
        """Return float32 depth or None while the readback ring is pending."""

    @overload
    def getRawImage(self, sensor: raisimpy.RGBCamera, mode: Camera.SensorStorageMode = Camera.SensorStorageMode.SENSOR_SETTER, flipVertical: bool = True) -> numpy.ndarray | None: ...

    @overload
    def getRawImage(self, sensor: raisimpy.DepthCamera, mode: Camera.SensorStorageMode = Camera.SensorStorageMode.SENSOR_SETTER, flipVertical: bool = True) -> numpy.ndarray | None: ...

    @overload
    def getRawImage(self, sensor: raisimpy.SegmentationCamera, mode: Camera.SensorStorageMode = Camera.SensorStorageMode.SENSOR_SETTER, flipVertical: bool = True) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None: ...

    def readSegmentation(self, flipVertical: bool = True) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Return int32 (objects, bodies, labels) images of the last renderSegmentation().
        """

class CameraFrustum:
    def __init__(self, viewer: RayraiWindow, visual: Visuals, mesh: OpenGLMesh) -> None: ...

    def updateFromCamera(self, camera: Camera) -> None:
        """
        /**
           * @brief Update frustum mesh from a rayrai camera.
           *
           * Handles both perspective (vertical FOV from Camera::zoom) and orthographic
           * (Camera::orthoScale) projections. The drawn far distance is clamped to
           * 10 m. Requires a current OpenGL context (the mesh is re-uploaded).
           * @param camera Source camera.
           */
        C++: void (const Camera &)
        """

    def updateFromDepthCamera(self, depthCam: raisimpy.DepthCamera) -> None:
        """
        /**
           * @brief Update frustum mesh from a raisim depth camera.
           *
           * Calls `depthCam.updatePose()` first. The sensor's horizontal FOV is read
           * as radians, or as degrees when it exceeds pi. The drawn far distance is
           * clamped to 10 m. Requires a current OpenGL context.
           * @param depthCam Source depth camera.
           */
        C++: void (raisim::DepthCamera &)
        """

    def updatePerspective(self, position: numpy.ndarray, orientation: numpy.ndarray, horizontalFov: float, aspect: float, nearDistance: float, farDistance: float) -> None:
        """
        /**
           * @brief Update a perspective frustum from an explicit world-space sensor pose.
           *
           * The sensor convention is local +X forward and local +Z up, matching RaiSim
           * RGB and depth cameras.
           *
           * @param position Sensor position in world coordinates.
           * @param orientation Sensor-to-world orientation (normalized internally;
           *        identity if degenerate).
           * @param horizontalFov Horizontal field of view in radians, clamped to
           *        (0, ~179 deg).
           * @param aspect Image width divided by image height.
           * @param nearDistance Near-plane distance.
           * @param farDistance Far-plane distance used to size the visualization.
           */
        C++: void (const glm::vec3 &, const glm::quat &, float, float, float, float)
        """

    def setColor(self, color: numpy.ndarray) -> None:
        """
        /**
           * @brief Set the frustum color.
           * @param color RGBA color.
           */
        C++: void (const glm::vec4 &)
        """

    def setDetectable(self, detectable: bool) -> None:
        """
        /**
           * @brief Set the capture/detection render category on the frustum visual.
           *
           * Camera frustums are usually viewer-only helpers and should normally remain
           * non-detectable so they do not appear in external-camera or generated
           * capture images. Set this to true only when the frustum helper itself should
           * be included by detectable-only visualization passes.
           *
           * @param detectable True to include the frustum visual in detectable-only
           *        render passes; false to keep it viewer-only for those passes.
           * @see VisualCategory
           */
        C++: void (bool)
        """

    def getVisual(self) -> Visuals:
        """
        /**
           * @brief Get the underlying Visuals object.
           * @return Shared pointer to the visual.
           */
        C++: std::shared_ptr<Visuals> () const
        """

class CloudQuality(enum.Enum):
    Off = 0

    Texture = 1

    Volumetric = 2

class ColorGradePreset(enum.Enum):
    Neutral = 0

    Warm = 1

    Cool = 2

    Cinematic = 3

    Bleach = 4

class CoordinateFrame:
    def __init__(self, viewer: RayraiWindow, asset: RayraiGlobalAsset) -> None: ...

    class Pose:
        def __init__(self) -> None: ...

        @property
        def position(self) -> numpy.ndarray: ...

        @position.setter
        def position(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def quaternion(self) -> numpy.ndarray: ...

        @quaternion.setter
        def quaternion(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def poses(self) -> list[CoordinateFrame.Pose]: ...

    @poses.setter
    def poses(self, arg: Sequence[CoordinateFrame.Pose], /) -> None: ...

    @property
    def frameSize(self) -> float: ...

    @frameSize.setter
    def frameSize(self, arg: float, /) -> None: ...

    @overload
    def draw(self, shader: Shader) -> None:
        """
        /**
           * @brief Draws all coordinate frames in #poses without culling.
           *
           * Each axis is drawn as a cylinder (X red, Y green, Z blue) by setting the
           * `model` and `objectColor` uniforms. The shader is not bound here, so
           * @p shader must already be in use. Requires a current OpenGL context.
           * @param shader The shader program to use for rendering.
           */
        C++: void (Shader &)
        """

    @overload
    def draw(self, shader: Shader, viewProjection: numpy.ndarray, boundShader: bool) -> None:
        """
        /**
           * @brief Variant that also frustum-culls poses against @p viewProjection.
           * @param shader Shader that must already be in use.
           * @param viewProjection Camera view-projection matrix; an all-zero matrix
           *        disables culling.
           * @param boundShader Unused; kept for API parity (neither overload binds
           *        the shader).
           */
        C++: void (Shader &, const glm::mat4 &, bool)
        """

    def enable(self, enable: bool) -> None:
        """
        /**
           * @brief Enables or disables rendering of this coordinate frame set.
           * @param enable True to enable, false to disable.
           */
        C++: void (bool)
        """

    def enabled(self) -> bool:
        """
        /// @return True when this frame set is rendered.
        C++: bool () const
        """

class ExplosionEffect:
    def __init__(self) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def chargeMass(self) -> float: ...

    @chargeMass.setter
    def chargeMass(self, arg: float, /) -> None: ...

    @property
    def onGround(self) -> bool: ...

    @onGround.setter
    def onGround(self, arg: bool, /) -> None: ...

    @property
    def smokeColor(self) -> numpy.ndarray: ...

    @smokeColor.setter
    def smokeColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def dustColor(self) -> numpy.ndarray: ...

    @dustColor.setter
    def dustColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def inWater(self) -> bool: ...

    @inWater.setter
    def inWater(self, arg: bool, /) -> None: ...

class ExplosionEffectSettings:
    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def sizeScale(self) -> float: ...

    @sizeScale.setter
    def sizeScale(self, arg: float, /) -> None: ...

    @property
    def smokeLifetime(self) -> float: ...

    @smokeLifetime.setter
    def smokeLifetime(self, arg: float, /) -> None: ...

    @property
    def marchSteps(self) -> int: ...

    @marchSteps.setter
    def marchSteps(self, arg: int, /) -> None: ...

    @property
    def volumeDownsample(self) -> int: ...

    @volumeDownsample.setter
    def volumeDownsample(self, arg: int, /) -> None: ...

    @property
    def sparks(self) -> bool: ...

    @sparks.setter
    def sparks(self, arg: bool, /) -> None: ...

    @property
    def flashLight(self) -> bool: ...

    @flashLight.setter
    def flashLight(self, arg: bool, /) -> None: ...

    @property
    def scorchMarks(self) -> bool: ...

    @scorchMarks.setter
    def scorchMarks(self, arg: bool, /) -> None: ...

    @property
    def scorchLifetime(self) -> float: ...

    @scorchLifetime.setter
    def scorchLifetime(self, arg: float, /) -> None: ...

    @property
    def tracers(self) -> bool: ...

    @tracers.setter
    def tracers(self, arg: bool, /) -> None: ...

    @property
    def useWorldTime(self) -> bool: ...

    @useWorldTime.setter
    def useWorldTime(self, arg: bool, /) -> None: ...

    @property
    def smokeColor(self) -> numpy.ndarray: ...

    @smokeColor.setter
    def smokeColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def dustColor(self) -> numpy.ndarray: ...

    @dustColor.setter
    def dustColor(self, arg: numpy.ndarray, /) -> None: ...

class FireEffect:
    def __init__(self) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def size(self) -> float: ...

    @size.setter
    def size(self, arg: float, /) -> None: ...

    @property
    def duration(self) -> float: ...

    @duration.setter
    def duration(self, arg: float, /) -> None: ...

    @property
    def smokeColor(self) -> numpy.ndarray: ...

    @smokeColor.setter
    def smokeColor(self, arg: numpy.ndarray, /) -> None: ...

class GeometryRefractionBackend(enum.Enum):
    Automatic = 0

    Portable = 1

    VulkanRayQuery = 2

class GeometryRefractionDiagnostics:
    def __init__(self) -> None: ...

    @property
    def triangles(self) -> int: ...

    @triangles.setter
    def triangles(self, arg: int, /) -> None: ...

    @property
    def volumes(self) -> int: ...

    @volumes.setter
    def volumes(self, arg: int, /) -> None: ...

    @property
    def bvhNodes(self) -> int: ...

    @bvhNodes.setter
    def bvhNodes(self, arg: int, /) -> None: ...

    @property
    def sceneBuilds(self) -> int: ...

    @sceneBuilds.setter
    def sceneBuilds(self, arg: int, /) -> None: ...

    @property
    def accumulatedSamples(self) -> int: ...

    @accumulatedSamples.setter
    def accumulatedSamples(self, arg: int, /) -> None: ...

    @property
    def activeCameraMedia(self) -> int: ...

    @activeCameraMedia.setter
    def activeCameraMedia(self, arg: int, /) -> None: ...

    @property
    def active(self) -> bool: ...

    @active.setter
    def active(self, arg: bool, /) -> None: ...

    @property
    def backend(self) -> GeometryRefractionBackend: ...

    @backend.setter
    def backend(self, arg: GeometryRefractionBackend, /) -> None: ...

    @property
    def backendStatus(self) -> str: ...

    @backendStatus.setter
    def backendStatus(self, arg: str, /) -> None: ...

class GeometryRefractionSamplingStatistics:
    def __init__(self) -> None: ...

    @property
    def pixels(self) -> int: ...

    @pixels.setter
    def pixels(self, arg: int, /) -> None: ...

    @property
    def belowBudget(self) -> int: ...

    @belowBudget.setter
    def belowBudget(self, arg: int, /) -> None: ...

    @property
    def meanSamples(self) -> float: ...

    @meanSamples.setter
    def meanSamples(self, arg: float, /) -> None: ...

    @property
    def minSamples(self) -> int: ...

    @minSamples.setter
    def minSamples(self, arg: int, /) -> None: ...

    @property
    def maxSamples(self) -> int: ...

    @maxSamples.setter
    def maxSamples(self, arg: int, /) -> None: ...

class GerstnerWave:
    def __init__(self) -> None: ...

    @property
    def direction(self) -> numpy.ndarray: ...

    @direction.setter
    def direction(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def wavelength(self) -> float: ...

    @wavelength.setter
    def wavelength(self, arg: float, /) -> None: ...

    @property
    def amplitude(self) -> float: ...

    @amplitude.setter
    def amplitude(self, arg: float, /) -> None: ...

    @property
    def steepness(self) -> float: ...

    @steepness.setter
    def steepness(self, arg: float, /) -> None: ...

    @property
    def phase(self) -> float: ...

    @phase.setter
    def phase(self, arg: float, /) -> None: ...

class GlowBlendMode(enum.Enum):
    Additive = 0

    Screen = 1

    Softlight = 2

    Replace = 3

    Mix = 4

class InstancedVisuals:
    @overload
    def __init__(self, viewer: RayraiWindow, type: raisimpy.ShapeType, name: str, size: numpy.ndarray, color1: numpy.ndarray, color2: numpy.ndarray, assets: RayraiGlobalAsset, meshFile: str = '') -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, type: raisimpy.ShapeType, name: str, size: numpy.ndarray, color1: numpy.ndarray, color2: numpy.ndarray, assets: RayraiGlobalAsset, meshFile: str, automaticMeshLodEnabled: bool) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, name: str, size: numpy.ndarray, color1: numpy.ndarray, color2: numpy.ndarray, meshes: Sequence[OpenGLMesh]) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, name: str, size: numpy.ndarray, color1: numpy.ndarray, color2: numpy.ndarray, meshes: Sequence[OpenGLMesh], automaticMeshLodEnabled: bool) -> None: ...

    class AsyncFoliageClusterPrepDiagnostics:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def applicable(self) -> bool: ...

        @applicable.setter
        def applicable(self, arg: bool, /) -> None: ...

        @property
        def scheduled(self) -> bool: ...

        @scheduled.setter
        def scheduled(self, arg: bool, /) -> None: ...

        @property
        def active(self) -> bool: ...

        @active.setter
        def active(self, arg: bool, /) -> None: ...

        @property
        def ready(self) -> bool: ...

        @ready.setter
        def ready(self, arg: bool, /) -> None: ...

        @property
        def committed(self) -> bool: ...

        @committed.setter
        def committed(self, arg: bool, /) -> None: ...

        @property
        def synchronousFallback(self) -> bool: ...

        @synchronousFallback.setter
        def synchronousFallback(self, arg: bool, /) -> None: ...

        @property
        def authoredInstanceCount(self) -> int: ...

        @authoredInstanceCount.setter
        def authoredInstanceCount(self, arg: int, /) -> None: ...

        @property
        def nodeCount(self) -> int: ...

        @nodeCount.setter
        def nodeCount(self, arg: int, /) -> None: ...

        @property
        def leafCount(self) -> int: ...

        @leafCount.setter
        def leafCount(self, arg: int, /) -> None: ...

        @property
        def fallbackCode(self) -> int: ...

        @fallbackCode.setter
        def fallbackCode(self, arg: int, /) -> None: ...

    class FoliageBatchSupport:
        def __init__(self) -> None: ...

        @property
        def basicTransform(self) -> bool: ...

        @basicTransform.setter
        def basicTransform(self, arg: bool, /) -> None: ...

        @property
        def colorBatch(self) -> bool: ...

        @colorBatch.setter
        def colorBatch(self, arg: bool, /) -> None: ...

    class FoliageImpostorDiagnostics:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def applicable(self) -> bool: ...

        @applicable.setter
        def applicable(self, arg: bool, /) -> None: ...

        @property
        def used(self) -> bool: ...

        @used.setter
        def used(self, arg: bool, /) -> None: ...

        @property
        def authoredInstanceCount(self) -> int: ...

        @authoredInstanceCount.setter
        def authoredInstanceCount(self, arg: int, /) -> None: ...

        @property
        def renderedInstanceCount(self) -> int: ...

        @renderedInstanceCount.setter
        def renderedInstanceCount(self, arg: int, /) -> None: ...

        @property
        def candidateInstanceCount(self) -> int: ...

        @candidateInstanceCount.setter
        def candidateInstanceCount(self, arg: int, /) -> None: ...

        @property
        def sourceTrianglesPerInstance(self) -> int: ...

        @sourceTrianglesPerInstance.setter
        def sourceTrianglesPerInstance(self, arg: int, /) -> None: ...

        @property
        def impostorTrianglesPerInstance(self) -> int: ...

        @impostorTrianglesPerInstance.setter
        def impostorTrianglesPerInstance(self, arg: int, /) -> None: ...

        @property
        def nearestDistance(self) -> float: ...

        @nearestDistance.setter
        def nearestDistance(self, arg: float, /) -> None: ...

        @property
        def projectedRadiusPixels(self) -> float: ...

        @projectedRadiusPixels.setter
        def projectedRadiusPixels(self, arg: float, /) -> None: ...

        @property
        def fallbackCode(self) -> int: ...

        @fallbackCode.setter
        def fallbackCode(self, arg: int, /) -> None: ...

    class FoliageWindDiagnostics:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def rootHeight(self) -> float: ...

        @rootHeight.setter
        def rootHeight(self, arg: float, /) -> None: ...

        @property
        def tipHeight(self) -> float: ...

        @tipHeight.setter
        def tipHeight(self, arg: float, /) -> None: ...

        @property
        def windStrength(self) -> float: ...

        @windStrength.setter
        def windStrength(self, arg: float, /) -> None: ...

        @property
        def stiffness(self) -> float: ...

        @stiffness.setter
        def stiffness(self, arg: float, /) -> None: ...

        @property
        def flutterWeight(self) -> float: ...

        @flutterWeight.setter
        def flutterWeight(self, arg: float, /) -> None: ...

    class GpuHizOcclusionConfig:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def depthPyramidTexture(self) -> int: ...

        @depthPyramidTexture.setter
        def depthPyramidTexture(self, arg: int, /) -> None: ...

        @property
        def width(self) -> int: ...

        @width.setter
        def width(self, arg: int, /) -> None: ...

        @property
        def height(self) -> int: ...

        @height.setter
        def height(self, arg: int, /) -> None: ...

        @property
        def mipLevels(self) -> int: ...

        @mipLevels.setter
        def mipLevels(self, arg: int, /) -> None: ...

        @property
        def depthEpsilon(self) -> float: ...

        @depthEpsilon.setter
        def depthEpsilon(self, arg: float, /) -> None: ...

    class GrassPatchDiagnostics:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def authoredInstanceCount(self) -> int: ...

        @authoredInstanceCount.setter
        def authoredInstanceCount(self, arg: int, /) -> None: ...

        @property
        def drawBudget(self) -> int: ...

        @drawBudget.setter
        def drawBudget(self, arg: int, /) -> None: ...

        @property
        def drawStride(self) -> int: ...

        @drawStride.setter
        def drawStride(self, arg: int, /) -> None: ...

        @property
        def patchWidth(self) -> float: ...

        @patchWidth.setter
        def patchWidth(self, arg: float, /) -> None: ...

        @property
        def patchDepth(self) -> float: ...

        @patchDepth.setter
        def patchDepth(self, arg: float, /) -> None: ...

        @property
        def densityPerSquareMeter(self) -> float: ...

        @densityPerSquareMeter.setter
        def densityPerSquareMeter(self, arg: float, /) -> None: ...

        @property
        def maxDrawDistance(self) -> float: ...

        @maxDrawDistance.setter
        def maxDrawDistance(self, arg: float, /) -> None: ...

        @property
        def lodFadeStartDistance(self) -> float: ...

        @lodFadeStartDistance.setter
        def lodFadeStartDistance(self, arg: float, /) -> None: ...

        @property
        def lodFadeEndDistance(self) -> float: ...

        @lodFadeEndDistance.setter
        def lodFadeEndDistance(self, arg: float, /) -> None: ...

        @property
        def bladeRootHeight(self) -> float: ...

        @bladeRootHeight.setter
        def bladeRootHeight(self, arg: float, /) -> None: ...

        @property
        def bladeTipHeight(self) -> float: ...

        @bladeTipHeight.setter
        def bladeTipHeight(self, arg: float, /) -> None: ...

        @property
        def windStrength(self) -> float: ...

        @windStrength.setter
        def windStrength(self, arg: float, /) -> None: ...

        @property
        def stiffness(self) -> float: ...

        @stiffness.setter
        def stiffness(self, arg: float, /) -> None: ...

        @property
        def flutterWeight(self) -> float: ...

        @flutterWeight.setter
        def flutterWeight(self, arg: float, /) -> None: ...

        @property
        def seed(self) -> int: ...

        @seed.setter
        def seed(self, arg: int, /) -> None: ...

    class HierarchicalFoliageClusterDiagnostics:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def applicable(self) -> bool: ...

        @applicable.setter
        def applicable(self, arg: bool, /) -> None: ...

        @property
        def used(self) -> bool: ...

        @used.setter
        def used(self, arg: bool, /) -> None: ...

        @property
        def authoredInstanceCount(self) -> int: ...

        @authoredInstanceCount.setter
        def authoredInstanceCount(self, arg: int, /) -> None: ...

        @property
        def renderedInstanceCount(self) -> int: ...

        @renderedInstanceCount.setter
        def renderedInstanceCount(self, arg: int, /) -> None: ...

        @property
        def candidateInstanceCount(self) -> int: ...

        @candidateInstanceCount.setter
        def candidateInstanceCount(self, arg: int, /) -> None: ...

        @property
        def visibleInstanceCount(self) -> int: ...

        @visibleInstanceCount.setter
        def visibleInstanceCount(self, arg: int, /) -> None: ...

        @property
        def nodeCount(self) -> int: ...

        @nodeCount.setter
        def nodeCount(self, arg: int, /) -> None: ...

        @property
        def leafCount(self) -> int: ...

        @leafCount.setter
        def leafCount(self, arg: int, /) -> None: ...

        @property
        def visitedNodeCount(self) -> int: ...

        @visitedNodeCount.setter
        def visitedNodeCount(self, arg: int, /) -> None: ...

        @property
        def visibleLeafCount(self) -> int: ...

        @visibleLeafCount.setter
        def visibleLeafCount(self, arg: int, /) -> None: ...

        @property
        def fallbackCode(self) -> int: ...

        @fallbackCode.setter
        def fallbackCode(self, arg: int, /) -> None: ...

    class InstanceSpec:
        def __init__(self) -> None: ...

        @property
        def position(self) -> numpy.ndarray: ...

        @position.setter
        def position(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def orientation(self) -> numpy.ndarray: ...

        @orientation.setter
        def orientation(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def scale(self) -> numpy.ndarray: ...

        @scale.setter
        def scale(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def colorWeight(self) -> float: ...

        @colorWeight.setter
        def colorWeight(self, arg: float, /) -> None: ...

    class ShadowFoliageLodDiagnostics:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def applicable(self) -> bool: ...

        @applicable.setter
        def applicable(self, arg: bool, /) -> None: ...

        @property
        def used(self) -> bool: ...

        @used.setter
        def used(self, arg: bool, /) -> None: ...

        @property
        def authoredInstanceCount(self) -> int: ...

        @authoredInstanceCount.setter
        def authoredInstanceCount(self, arg: int, /) -> None: ...

        @property
        def renderedInstanceCount(self) -> int: ...

        @renderedInstanceCount.setter
        def renderedInstanceCount(self, arg: int, /) -> None: ...

        @property
        def candidateInstanceCount(self) -> int: ...

        @candidateInstanceCount.setter
        def candidateInstanceCount(self, arg: int, /) -> None: ...

        @property
        def drawStride(self) -> int: ...

        @drawStride.setter
        def drawStride(self, arg: int, /) -> None: ...

        @property
        def fallbackCode(self) -> int: ...

        @fallbackCode.setter
        def fallbackCode(self, arg: int, /) -> None: ...

    def cloneWithoutInstances(self, name: str) -> InstancedVisuals:
        """
        /// Share prepared geometry, LODs and appearance with a new empty batch.
          /// Instance storage and GPU buffers remain independent. The source must
          /// have completed asynchronous preparation; use on its owning GL context.
          /// Meshes and materials are shared: edit instance appearance on the batch,
          /// and treat the prepared mesh/material objects as immutable while shared.
          /// @param name Name of the new batch; must be non-empty.
          /// @return The new, empty batch.
          /// @throws std::invalid_argument if @p name is empty or this is not a ready mesh batch.
        C++: std::shared_ptr<InstancedVisuals> (const std::string &) const
        """

    def count(self) -> int:
        """
        /**
           * @brief Get number of instances.
           * @return Instance count.
           */
        C++: size_t () const
        """

    def getName(self) -> str:
        """
        /// @return Batch name.
        C++: const std::string &() const
        """

    def clearInstances(self) -> None:
        """
        /**
           * @brief Clear all instances.
           */
        C++: void ()
        """

    def resize(self, count: int) -> None:
        """
        /**
           * @brief Resize instance storage.
           *
           * Every instance (existing ones included) is reset to the origin, identity
           * orientation, baseScale() and color weight 0.
           * @param count New instance count.
           */
        C++: void (size_t)
        """

    def reserveInstances(self, capacity: int) -> None:
        """
        /**
           * @brief Reserve instance storage without changing the authored instance count.
           * @param capacity Number of instances to reserve room for.
           */
        C++: void (size_t)
        """

    @overload
    def addInstance(self, pos: numpy.ndarray, ori: numpy.ndarray, scale: numpy.ndarray, colorWeight: float = 0.0) -> None:
        """
        /**
           * @brief Add a fully specified instance.
           * @param pos Position in world coordinates (m).
           * @param ori Orientation quaternion (wxyz): ori.x is w and ori.w is z.
           * @param scale Per-instance scale, multiplied by baseScale().
           * @param colorWeight Blend between color1 (0) and color2 (1).
           */
        C++: void (const glm::vec3 &, const glm::vec4 &, const glm::vec3 &, float)
        """

    @overload
    def addInstance(self, pos: numpy.ndarray, ori: numpy.ndarray, colorWeight: float = 0.0) -> None:
        """
        /**
           * @brief Add an instance with default scale (baseScale()).
           * @param pos Position.
           * @param ori Orientation quaternion (wxyz).
           * @param colorWeight Blend between color1 and color2.
           */
        C++: void (const glm::vec3 &, const glm::vec4 &, float)
        """

    @overload
    def addInstance(self, pos: numpy.ndarray, colorWeight: float = 0.0) -> None:
        """
        /**
           * @brief Add an instance at position with identity orientation and baseScale().
           * @param pos Position.
           * @param colorWeight Blend between color1 and color2.
           */
        C++: void (const glm::vec3 &, float)
        """

    def shapeType(self) -> raisimpy.ShapeType:
        """
        /// @return Shape type of the batch.
        C++: raisim::Shape::Type () const
        """

    def setPosition(self, id: int, pos: numpy.ndarray) -> None:
        """
        /**
           * @brief Update position of an instance (ignored for an out-of-range @p id).
           * @param id Instance index.
           * @param pos New position.
           */
        C++: void (size_t, const glm::vec3 &)
        """

    def setOrientation(self, id: int, ori: numpy.ndarray) -> None:
        """
        /**
           * @brief Update orientation of an instance.
           * @param id Instance index.
           * @param ori Orientation quaternion (wxyz).
           */
        C++: void (size_t, const glm::vec4 &)
        """

    def setScale(self, id: int, scale: numpy.ndarray) -> None:
        """
        /**
           * @brief Update scale of an instance (ignored for an out-of-range @p id).
           *
           * Unlike addInstance(), @p scale is stored as the instance's total scale and
           * is not multiplied by baseScale(); it matches what getScale() returns.
           * @param id Instance index.
           * @param scale New total scale.
           */
        C++: void (size_t, const glm::vec3 &)
        """

    def setColor(self, color1: numpy.ndarray, color2: numpy.ndarray) -> None:
        """
        /**
           * @brief Update the two base colors used for blending.
           * @param color1 Primary linear RGBA color in [0, 1].
           * @param color2 Secondary linear RGBA color in [0, 1].
           */
        C++: void (const glm::vec4 &, const glm::vec4 &)
        """

    def setUseMeshColor(self, use: bool) -> None:
        """
        /**
           * @brief Use mesh-authored colors for mesh instances instead of instance colors.
           * @param use True to use mesh colors (Mesh batches only).
           */
        C++: void (bool)
        """

    def usesMeshColor(self) -> bool:
        """
        /**
           * @return True when mesh-authored colors are requested.
           */
        C++: bool () const
        """

    def setDefaultPrimitivePbrEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Toggle High/Ultra default PBR shading for built-in primitive meshes.
           *
           * Edits the shared primitive mesh material, affecting every user of that primitive.
           * @param enabled True to allow the PBR pass.
           */
        C++: void (bool)
        """

    def setColorWeight(self, id: int, weight: float) -> None:
        """
        /**
           * @brief Set per-instance blend weight.
           * @param id Instance index.
           * @param weight Blend weight (0..1).
           */
        C++: void (size_t, float)
        """

    def setCategory(self, category: VisualCategory) -> None:
        """
        /**
           * @brief Set the render category used by capture/detection filters.
           *
           * The category is shared by all instances in this batch. It controls whether
           * the instanced visual is included when a rayrai pass excludes non-detectable
           * visualization objects, such as external camera scene-color renders, capture
           * paths that reuse external-camera rendering, or picking/shadow candidate
           * policies that request detectable custom visuals. The normal interactive
           * viewer includes both categories by default.
           *
           * This is render metadata only; it does not create collision, physics
           * behavior, or per-instance semantic labels. New `InstancedVisuals` default
           * to `VisualCategory::NotDetectable`.
           *
           * @param category New render category for the whole instanced batch.
           * @see VisualCategory
           */
        C++: void (VisualCategory)
        """

    def setDetectable(self, detectable: bool) -> None:
        """
        /**
           * @brief Convenience helper for toggling the instanced visual render category.
           * @param detectable True to include this batch in detectable-only
           *        capture/detection render passes; false to keep it viewer-only for
           *        those passes.
           */
        C++: void (bool)
        """

    def setSegmentationIds(self, objectIndex: int, bodyIndex: int, label: int) -> None:
        """
        /**
           * @brief Set the ids a segmentation render writes for every instance
           *        (RayraiWindow::renderSegmentation() with visualization objects).
           *
           * Defaults to raisim::SegmentationCamera::kNotWorldObject (-2) as object and body index and to
           * label 0.
           * @param objectIndex Object image value.
           * @param bodyIndex Body image value.
           * @param label Label image value.
           */
        C++: void (int32_t, int32_t, int32_t)
        """

    def setSegmentationLabel(self, label: int) -> None:
        """
        /**
           * @brief Set the label image value of every instance (see setSegmentationIds()).
           * @param label Label image value.
           */
        C++: void (int32_t)
        """

    def getSegmentationIds(self) -> numpy.ndarray:
        """
        /// @return (object index, body index, label, 0) written by segmentation renders.
        C++: glm::ivec4 () const
        """

    def setSegmentationExcluded(self, excluded: bool) -> None:
        """
        /**
           * @brief Leave this visual out of segmentation renders, e.g. a batch whose instances are also
           *        drawn there one by one with ids of their own.
           * @param excluded True to skip this visual in segmentation renders.
           */
        C++: void (bool)
        """

    def isSegmentationExcluded(self) -> bool:
        """
        /// @return True when segmentation renders skip this visual.
        C++: bool () const
        """

    def getPosition(self, id: int) -> numpy.ndarray:
        """
        /**
           * @brief Get instance position.
           * @param id Instance index.
           * @return Position vector, or zero for an out-of-range @p id.
           */
        C++: glm::vec3 (size_t) const
        """

    def getOrientation(self, id: int) -> numpy.ndarray:
        """
        /**
           * @brief Get instance orientation (wxyz).
           * @param id Instance index.
           * @return Quaternion vector, or all zeros for an out-of-range @p id.
           */
        C++: glm::vec4 (size_t) const
        """

    def getScale(self, id: int) -> numpy.ndarray:
        """
        /**
           * @brief Get instance scale.
           * @param id Instance index.
           * @return Total scale (base scale already applied), or zero for an out-of-range @p id.
           */
        C++: glm::vec3 (size_t) const
        """

    def getColorWeight(self, id: int) -> float:
        """
        /**
           * @brief Get per-instance color blend weight.
           * @param id Instance index.
           * @return Weight, or 0 for an out-of-range @p id.
           */
        C++: float (size_t) const
        """

    def baseScale(self) -> numpy.ndarray:
        """
        /**
           * @return Base scale (the constructor's @p size) applied by addInstance() and resize().
           */
        C++: const glm::vec3 &() const
        """

    def getColor1(self) -> numpy.ndarray:
        """
        /**
           * @brief Get primary color.
           * @return Primary color.
           */
        C++: const glm::vec4 &() const
        """

    def getColor2(self) -> numpy.ndarray:
        """
        /**
           * @brief Get secondary color.
           * @return Secondary color.
           */
        C++: const glm::vec4 &() const
        """

    def getCategory(self) -> VisualCategory:
        """
        /**
           * @brief Get the render category used by capture/detection filters.
           * @return Current category for the whole instanced batch.
           */
        C++: VisualCategory () const
        """

    def isDetectable(self) -> bool:
        """
        /**
           * @brief Check whether detectable-only render passes include this batch.
           * @return True when the category is `VisualCategory::Detectable`.
           */
        C++: bool () const
        """

    def isTransparent(self) -> bool:
        """
        /**
           * @brief Check if any color is translucent.
           * @return True if either batch color has alpha < 1 or a mesh material needs the transparent pass.
           */
        C++: bool () const
        """

    def hasSpecialBlendMode(self) -> bool:
        """
        /**
           * @return True when at least one mesh uses a blend mode other than Material::BlendMode::Mix.
           */
        C++: bool () const
        """

    def hasPlanarReflectionMeshes(self) -> bool:
        """
        /// @return True when any PBR mesh uses planar reflection.
        C++: bool () const
        """

    def renderPriority(self) -> int:
        """
        /**
           * @return Highest material render priority across the instanced visual's meshes.
           */
        C++: int () const
        """

    def setCastsShadows(self, castsShadows: bool) -> None:
        """
        /**
           * @brief Set whether this instanced visual casts shadows (enabled by default).
           * Foliage materials and grass-patch configuration preserve this setting.
           * @param castsShadows True to render the batch into shadow maps.
           */
        C++: void (bool)
        """

    def castsShadows(self) -> bool:
        """
        /**
           * @return True when this instanced visual should cast shadows.
           */
        C++: bool () const
        """

    def isValid(self) -> bool:
        """
        /**
           * @brief Check if meshes are valid.
           * @return True if valid.
           */
        C++: bool () const
        """

    def renderRevision(self) -> int:
        """
        /// Internal: @return Counter bumped on render-affecting changes; used to invalidate frame caches.
        C++: uint64_t () const
        """

    def approximateBounds(self) -> tuple:
        """
        /**
           * @brief Bounding sphere for all instances.
           *
           * Each instance contributes a sphere at its position with radius equal to
           * the length of its scale, so the result is only conservative for source
           * geometry within the unit sphere.
           * @param[out] center Output center.
           * @param[out] radius Output radius.
           * @return True if there are instances to bound.
           */
        C++: bool (glm::vec3 &, float &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def shadowGeometryWithinBounds(self) -> bool:
        """
        /// @return True when shadow depth draws keep every vertex inside approximateBounds():
          ///         the source mesh fits the unit sphere, no material uses billboards,
          ///         fixed-size scaling or height displacement, and neither wind nor the
          ///         impostor policy applies.
        C++: bool () const
        """

    def approximateAabb(self) -> tuple:
        """
        /// Axis-aligned box version of approximateBounds() (same per-instance padding).
          /// @param[out] minCorner Minimum corner in world coordinates.
          /// @param[out] maxCorner Maximum corner in world coordinates.
          /// @return True if there are instances to bound.
        C++: bool (glm::vec3 &, glm::vec3 &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def sourceMeshLocalAabb(self) -> tuple:
        """
        /**
           * @brief Source mesh local-space AABB before per-instance transform.
           * @param[out] minCorner Minimum corner in mesh coordinates.
           * @param[out] maxCorner Maximum corner in mesh coordinates.
           * @return False when there is no mesh or no usable vertex data.
           */
        C++: bool (glm::vec3 &, glm::vec3 &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def setMaxRenderedInstances(self, maxInstances: int) -> None:
        """
        /**
           * @brief Cap drawn instances for simple single-threaded LOD. 0 means draw all.
           * @param maxInstances Only the first @p maxInstances instances are drawn.
           */
        C++: void (size_t)
        """

    def maxRenderedInstances(self) -> int:
        """
        /// @return Instance cap (0 = unlimited).
        C++: size_t () const
        """

    def setRenderedInstanceStride(self, stride: int) -> None:
        """
        /**
           * @brief Set a stride for simple per-camera-independent LOD. 1 draws every instance.
           * @param stride Draw every @p stride-th instance; 0 is treated as 1.
           */
        C++: void (size_t)
        """

    def renderedInstanceStride(self) -> int:
        """
        /// @return User-requested instance stride (>= 1).
        C++: size_t () const
        """

    def setFoliageDistanceFade(self, start: float, end: float, fadeIn: bool) -> None:
        """
        /// Fade this batch across a camera-distance band. The complementary in/out
          /// modes let two representations overlap without a hard cell LOD switch.
          /// Zero (the default) disables distance fading.
          /// @param start Camera distance where the fade band starts (m, > 0).
          /// @param end Camera distance where it ends (m, > @p start).
          /// @param fadeIn True to fade in across the band, false to fade out.
          /// @throws std::invalid_argument for non-finite or out-of-order values.
        C++: void (float, float, bool)
        """

    def clearFoliageDistanceFade(self) -> None:
        """
        /// Disable the distance fade set by setFoliageDistanceFade().
        C++: void ()
        """

    def setProjectedLodPolicy(self, enabled: bool, minProjectedRadiusPixels: float = 2.0, maxStride: int = 8) -> None:
        """
        /**
           * @brief Enable camera-projected LOD for instanced visuals.
           *
           * When enabled, the renderer increases the effective draw stride for far-away instances whose
           * approximate per-instance radius projects below minProjectedRadiusPixels. The explicit rendered
           * stride remains a lower bound, so user-requested LOD is preserved.
           * Requires a camera-aware draw() overload (a positive viewport height).
           * @param enabled Enable the policy.
           * @param minProjectedRadiusPixels Target projected instance radius in pixels (min 0.25).
           * @param maxStride Largest stride the policy may choose (min 1).
           */
        C++: void (bool, float, size_t)
        """

    def projectedLodEnabled(self) -> bool:
        """
        /// @return True when projected LOD is enabled.
        C++: bool () const
        """

    def projectedLodMinRadiusPixels(self) -> float:
        """
        /// @return Projected-LOD target radius in pixels.
        C++: float () const
        """

    def projectedLodMaxStride(self) -> int:
        """
        /// @return Projected-LOD maximum stride.
        C++: size_t () const
        """

    def setAutomaticMeshLodEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Enable automatic mesh LOD generation/selection for instanced mesh visuals.
           * @param enabled True to build (or fetch cached) LODs; false drops them.
           */
        C++: void (bool)
        """

    def setFoliageMeshLodPixelError(self, pixelError: float) -> None:
        """
        /// Set the projected geometric-error budget for authored foliage LODs.
          /// Smaller values keep finer meshes until they are farther from the camera.
          /// Zero restores the renderer default.
          /// @param pixelError Error budget in pixels, in [0, 8].
          /// @throws std::invalid_argument outside [0, 8] or for non-finite values.
        C++: void (float)
        """

    def foliageMeshLodPixelError(self) -> float:
        """
        /// @return Foliage LOD pixel-error budget (0 = renderer default).
        C++: float () const
        """

    def setFoliageShadowMeshLodPixelError(self, pixelError: float) -> None:
        """
        /// Set the projected geometric-error budget of the foliage LODs drawn into shadow maps, where
          /// coarser meshes rarely show. Zero leaves shadows to setFoliageMeshLodPixelError(), or to the
          /// renderer's shadow default when that is zero too.
          /// @param pixelError Error budget in pixels, in [0, 8].
          /// @throws std::invalid_argument outside [0, 8] or for non-finite values.
        C++: void (float)
        """

    def foliageShadowMeshLodPixelError(self) -> float:
        """
        /// @return Shadow foliage LOD pixel-error budget (0 = as the color budget).
        C++: float () const
        """

    @overload
    def pollAsyncMeshPreparation(self) -> bool:
        """
        /// Finish worker-prepared LODs incrementally on the render thread.
          /// Starts the worker on the first call and uploads one prepared submesh per
          /// later call; draws also call it. Requires the batch's GL context: once its
          /// uploads began in one context, a call with another current gives the
          /// staged uploads back (see releaseStagedAsyncMeshUploads()) and throws
          /// std::logic_error; a draw there leaves the batch pending instead.
          /// @return True on the call that installs the finished LODs.
        C++: bool ()
        """

    @overload
    def pollAsyncMeshPreparation(self, deadline: datetime.timedelta | float) -> bool:
        """
        /// Like pollAsyncMeshPreparation(), but uploads prepared submeshes until
          /// @p deadline passes (at least one per call).
          /// @param deadline Time after which no further upload starts.
          /// @return True on the call that installs the finished LODs.
        C++: bool (std::chrono::steady_clock::time_point)
        """

    def hasPendingAsyncMeshPreparation(self) -> bool:
        """
        /// @return True while automatic LODs are still being prepared (the batch is not drawn).
        C++: bool () const
        """

    def releaseStagedAsyncMeshUploads(self) -> None:
        """
        /// Gives the staging space of the prepared submeshes waiting for upload back,
          /// from any context; they stage again, or upload directly, once polled in
          /// the batch's context. RayraiWindow calls it when polled in another context.
        C++: void ()
        """

    def automaticMeshLodEnabled(self) -> bool:
        """
        /**
           * @return True when automatic mesh LOD is enabled for instanced mesh visuals.
           */
        C++: bool () const
        """

    def automaticMeshLodLevelCount(self) -> int:
        """
        /**
           * @return Number of available instanced mesh LOD levels including the source mesh.
           */
        C++: size_t () const
        """

    def individualFoliageInstanceUploadCount(self) -> int:
        """
        /// Internal (diagnostics): @return Instance uploads made by the per-instance foliage LOD path.
        C++: size_t () const
        """

    def individualFoliageLodSelectionCount(self) -> int:
        """
        /// Internal (diagnostics): @return Number of submesh/pass visibility and LOD classifications (cache misses).
        C++: size_t () const
        """

    def sourceMeshCountForDiagnostics(self) -> int:
        """
        /**
           * @return Source mesh count for renderer regressions and diagnostics.
           */
        C++: size_t () const
        """

    def sourceMeshCacheKey(self) -> str:
        """
        /**
           * @return Canonical source mesh cache key, empty for built-in primitive batches.
           */
        C++: const std::string &() const
        """

    def instanceAttributesBoundForDiagnostics(self) -> bool:
        """
        /**
           * @return Whether per-instance vertex attributes are bound to the current mesh VAOs (diagnostics).
           */
        C++: bool () const
        """

    def setPreparedMeshLods(self, levels: Sequence[Sequence[OpenGLMesh]]) -> None:
        """
        /// Install six authored LOD levels; source geometry remains level zero.
          /// Mesh lists may have different material/part counts. Every mesh must carry
          /// a finite nonnegative generatedLodRelativeError for projected selection.
          /// Authored levels use opaque/masked single-pass ordinary materials.
          /// Call on the GL render thread after loading; keep external textures alive.
          /// Replacement is atomic on validation failure. Disabling automatic LODs
          /// clears this chain; reinstall it to re-enable these authored levels.
          /// @param levels Exactly six mesh lists (LOD 1..6).
          /// @throws std::invalid_argument if the batch is not a ready mesh batch or a level is invalid.
        C++: void (std::span<const std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>>>)
        """

    def preparedMeshLod(self, level: int) -> list[OpenGLMesh]:
        """
        /// Ready geometry for building static proxies. Shared meshes are immutable;
          /// the prototype and its GL context must outlive their use by the caller.
          /// @param level LOD level; 0 or an unavailable level returns the source meshes.
          /// @return Mesh list of that level.
        C++: const std::vector<std::shared_ptr<OpenGLMesh>> &(size_t) const
        """

    def automaticMeshLodTriangleCount(self, level: int) -> int:
        """
        /**
           * @param level LOD level; 0 is the source mesh.
           * @return Triangle count for one instanced mesh LOD level, or 0 for an invalid level.
           */
        C++: size_t (size_t) const
        """

    def automaticMeshLodMaxUvClusterSpan(self, level: int) -> float:
        """
        /**
           * @param level Generated LOD level (>= 1).
           * @return Largest UV span collapsed into one generated LOD vertex (diagnostics).
           */
        C++: float (size_t) const
        """

    def selectAutomaticMeshLodLevel(self, viewPos: numpy.ndarray, verticalFovDeg: float, viewportHeight: int) -> int:
        """
        /**
           * @brief Select the instanced mesh LOD level that would be used for a camera.
           * @param viewPos Camera position in world coordinates.
           * @param verticalFovDeg Camera vertical field of view in degrees.
           * @param viewportHeight Viewport height in pixels.
           * @return Selected level (0 = source mesh).
           */
        C++: size_t (const glm::vec3 &, float, int) const
        """

    def configureGrassPatch(self, patchWidth: float, patchDepth: float, densityPerSquareMeter: float, drawBudget: int, maxDrawDistance: float, lodFadeStartDistance: float, lodFadeEndDistance: float, seed: int, bladeRootHeight: float = 0.0, bladeTipHeight: float = 1.0, windStrength: float = 1.0, stiffness: float = 0.3499999940395355, flutterWeight: float = 0.15000000596046448) -> None:
        """
        /**
           * @brief Mark this instanced visual as a grass patch and configure bounded draw/wind metadata.
           *
           * Instances are still authored through addInstance()/resize() so callers keep full control over
           * density masks and scatter placement. This method wires the batch into the generic foliage wind
           * shader path, projected LOD, and render diagnostics.
           *
           * Also caps drawn instances at @p drawBudget and enables projected LOD,
           * shadow foliage LOD and hierarchical cluster culling with fixed defaults.
           * Width, depth, density and seed are recorded for diagnostics only.
           * @param patchWidth Patch width (m).
           * @param patchDepth Patch depth (m).
           * @param densityPerSquareMeter Blade density.
           * @param drawBudget Maximum drawn instances (0 = unlimited).
           * @param maxDrawDistance View depth at which blades have faded out (m); 0 = no limit.
           * @param lodFadeStartDistance View depth where blades start fading (m).
           * @param lodFadeEndDistance View depth where that fade ends (m).
           * @param seed Scatter seed (metadata).
           * @param bladeRootHeight Local height below which blades do not bend.
           * @param bladeTipHeight Local height of full bend.
           * @param windStrength Wind bend strength (>= 0).
           * @param stiffness Bend stiffness, clamped to [0, 1].
           * @param flutterWeight High-frequency flutter weight (>= 0).
           */
        C++: void (float, float, float, size_t, float, float, float, uint32_t, float, float, float, float, float)
        """

    def configureFoliageWind(self, rootHeight: float, tipHeight: float, windStrength: float, stiffness: float = 0.5, flutterWeight: float = 0.15000000596046448) -> None:
        """
        /**
           * @brief Configure generic wind bending for instanced foliage without grass-patch LOD.
           *
           * Also enables impostors, shadow foliage LOD and hierarchical cluster culling.
           * @param rootHeight Local height below which geometry does not bend.
           * @param tipHeight Local height of full bend.
           * @param windStrength Wind bend strength (>= 0).
           * @param stiffness Bend stiffness, clamped to [0, 1].
           * @param flutterWeight High-frequency flutter weight (>= 0).
           */
        C++: void (float, float, float, float, float)
        """

    def setShadowFoliageLodPolicy(self, enabled: bool, minProjectedRadiusPixels: float = 2.5, maxStride: int = 16, maxShadowDistance: float = 0.0) -> None:
        """
        /**
           * @brief Configure a shadow-only foliage instance LOD policy.
           *
           * This does not change the visible color pass. It only raises the instance
           * stride used while rendering foliage into shadow maps.
           * @param enabled Enable the policy.
           * @param minProjectedRadiusPixels Target projected instance radius in pixels (min 0.25).
           * @param maxStride Largest shadow stride (min 1).
           * @param maxShadowDistance Skip the batch in shadow passes when its nearest
           *        point is farther from the camera than this (m); 0 = no limit.
           */
        C++: void (bool, float, size_t, float)
        """

    def shadowFoliageLodPolicyEnabled(self) -> bool:
        """
        /// @return True when the shadow foliage LOD policy is enabled.
        C++: bool () const
        """

    def setFoliageImpostorPolicy(self, enabled: bool, maxProjectedRadiusPixels: float = 1.149999976158142, minInstanceCount: int = 1024) -> None:
        """
        /**
           * @brief Configure distant foliage impostor rendering for instanced mesh foliage.
           *
           * The visible color pass switches to a generated camera-facing quad only when the
           * source mesh projects below maxProjectedRadiusPixels. This keeps close/mid-range
           * authored foliage unchanged while preserving the authored instance count.
           * @param enabled Enable the policy.
           * @param maxProjectedRadiusPixels Switch to impostors below this projected radius (min 0.05).
           * @param minInstanceCount Only batches with at least this many instances qualify (min 1).
           */
        C++: void (bool, float, size_t)
        """

    def foliageImpostorPolicyEnabled(self) -> bool:
        """
        /// @return True when the foliage impostor policy is enabled.
        C++: bool () const
        """

    def setHierarchicalFoliageClustersPolicy(self, enabled: bool, leafTargetInstances: int = 32, minInstanceCount: int = 256, minRejectionRatio: float = 0.07999999821186066) -> None:
        """
        /**
           * @brief Configure conservative BVH/quadtree-style foliage cluster culling.
           *
           * Cluster culling only removes instances whose cluster bounds are outside the
           * current view/shadow frustum. It does not reduce authored instance count or
           * change material/mesh quality.
           * @param enabled Enable the policy.
           * @param leafTargetInstances Target instances per leaf, clamped to [8, 1024].
           * @param minInstanceCount Only batches with at least this many instances qualify (min 1).
           * @param minRejectionRatio Minimum culled fraction for the cluster path to be used, clamped to [0, 0.95].
           */
        C++: void (bool, size_t, size_t, float)
        """

    def hierarchicalFoliageClustersEnabled(self) -> bool:
        """
        /// @return True when hierarchical cluster culling is enabled.
        C++: bool () const
        """

    def hasFoliageShadowLodMaterialForDiagnostics(self) -> bool:
        """
        /// Internal: @return True when the batch is treated as foliage for shadow LOD (wind/grass configured or foliage-like materials).
        C++: bool () const
        """

    def supportsBasicFoliageShadowTransform(self) -> bool:
        """
        /** @return True when every possible shadow mesh uses ordinary instance transforms. */
        C++: bool () const
        """

    def supportsFoliageColorBatch(self) -> bool:
        """
        /// Internal: @return True when the batch can join the renderer's batched foliage color draw.
        C++: bool () const
        """

    def foliageBatchSupport(self) -> InstancedVisuals.FoliageBatchSupport:
        """C++: FoliageBatchSupport () const"""

    def hasDepthWritingMaterial(self) -> bool:
        """
        /// @return Whether a draw can override a surrounding pass's disabled depth writes.
        C++: bool () const
        """

    def usesDefaultDepthAndStencilState(self) -> bool:
        """
        /// @return True when every mesh material uses default depth/stencil state.
        C++: bool () const
        """

    def hasAlphaMaskedFoliageMaterialForDiagnostics(self) -> bool:
        """
        /// Internal (diagnostics): @return True when any mesh (source, LOD or impostor) uses an alpha-masked material.
        C++: bool () const
        """

    def grassPatchDiagnostics(self) -> InstancedVisuals.GrassPatchDiagnostics:
        """
        /// Internal (diagnostics): @return Grass-patch settings.
        C++: const GrassPatchDiagnostics &() const
        """

    def foliageWindDiagnostics(self) -> InstancedVisuals.FoliageWindDiagnostics:
        """
        /// Internal (diagnostics): @return Foliage wind settings.
        C++: const FoliageWindDiagnostics &() const
        """

    def shadowFoliageLodDiagnostics(self) -> InstancedVisuals.ShadowFoliageLodDiagnostics:
        """
        /// Internal (diagnostics): @return Last shadow foliage LOD decision.
        C++: const ShadowFoliageLodDiagnostics &() const
        """

    def lastShadowFoliageLodFallbackReasonForDiagnostics(self) -> int | None:
        """
        /// Internal (diagnostics): @return Text for ShadowFoliageLodDiagnostics::fallbackCode.
        C++: const char *() const
        """

    def foliageImpostorDiagnostics(self) -> InstancedVisuals.FoliageImpostorDiagnostics:
        """
        /// Internal (diagnostics): @return Last foliage impostor decision.
        C++: const FoliageImpostorDiagnostics &() const
        """

    def lastFoliageImpostorFallbackReasonForDiagnostics(self) -> int | None:
        """
        /// Internal (diagnostics): @return Text for FoliageImpostorDiagnostics::fallbackCode.
        C++: const char *() const
        """

    def hierarchicalFoliageClusterDiagnostics(self) -> InstancedVisuals.HierarchicalFoliageClusterDiagnostics:
        """
        /// Internal (diagnostics): @return Last cluster culling result.
        C++: const HierarchicalFoliageClusterDiagnostics &() const
        """

    def lastHierarchicalFoliageClusterFallbackReasonForDiagnostics(self) -> int | None:
        """
        /// Internal (diagnostics): @return Text for HierarchicalFoliageClusterDiagnostics::fallbackCode.
        C++: const char *() const
        """

    def requestAsyncFoliageClusterPreparation(self) -> None:
        """
        /// Start building the foliage cluster tree on a worker thread when the batch
          /// qualifies and the tree is out of date; no-op otherwise.
        C++: void ()
        """

    def pollAsyncFoliageClusterPreparation(self) -> bool:
        """
        /// Non-blocking check for a finished worker build.
          /// @return True when a result was installed by this call.
        C++: bool ()
        """

    def asyncFoliageClusterPrepDiagnostics(self) -> InstancedVisuals.AsyncFoliageClusterPrepDiagnostics:
        """
        /// Internal (diagnostics): @return Asynchronous cluster preparation state.
        C++: const AsyncFoliageClusterPrepDiagnostics &() const
        """

    def setDoubleBufferedInstanceUploads(self, enabled: bool) -> None:
        """
        /**
           * @brief Enable double-buffering for full instance-buffer uploads.
           *
           * Dirty-range updates remain on the single-buffer subdata path. Double-buffering helps when
           * full rebuilds are frequent, for example after sorting or projected LOD stride changes.
           * @param enabled True to alternate between two instance buffers.
           */
        C++: void (bool)
        """

    def setSortTransparentInstances(self, enabled: bool) -> None:
        """
        /**
           * @brief Enable optional back-to-front CPU sorting for transparent instances.
           *
           * This is disabled by default to keep RL visual workloads on the fast path.
           * @param enabled True to let the renderer call sortInstancesBackToFront() for this batch.
           */
        C++: void (bool)
        """

    def sortsTransparentInstances(self) -> bool:
        """
        /// @return True when transparent instance sorting is enabled.
        C++: bool () const
        """

    def sortInstancesBackToFront(self, viewPos: numpy.ndarray) -> None:
        """
        /// Reorder the instances farthest-first by distance from @p viewPos (stable).
          /// No-op unless setSortTransparentInstances(true), or when the instances are
          /// already in that order. Instance indices change.
          /// @param viewPos Camera position in world coordinates.
        C++: void (const glm::vec3 &)
        """

    def applyTextureFilteringQuality(self, anisotropy: float, mipLodBias: float) -> int:
        """
        /**
           * @brief Apply sampler quality to all 2D material textures in the instanced mesh.
           *
           * Modifies the shared GL texture objects.
           * @param anisotropy Requested anisotropic filtering level.
           * @param mipLodBias Mip LOD bias.
           * @return Number of texture ids updated (summed per mesh).
           */
        C++: size_t (float, float) const
        """

    def collectMaterialTextureIds(self) -> tuple:
        """
        /**
           * @brief Append non-environment material texture ids used by this instanced visual.
           * @param[in,out] textureIds Ids are appended (duplicates are kept).
           */
        C++: void (std::vector<unsigned int> &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def accumulateGeometryDiagnostics(self, diagnostics: OpenGLMesh.GeometryDiagnostics, tinyTriangleAreaThreshold: float) -> None:
        """
        /**
           * @brief Accumulate CPU-side mesh geometry diagnostics for this instanced visual.
           * @param[in,out] diagnostics Totals to add to.
           * @param tinyTriangleAreaThreshold Triangles with smaller (non-zero) area count as tiny.
           */
        C++: void (OpenGLMesh::GeometryDiagnostics &, float) const
        """

    def collectMeshDiagnostics(self, tinyTriangleAreaThreshold: float) -> tuple:
        """
        /**
           * @brief Append per-mesh geometry/material diagnostics for this instanced visual.
           * @param[in,out] diagnostics One entry is appended per source mesh.
           * @param tinyTriangleAreaThreshold Triangles with smaller (non-zero) area count as tiny.
           */
        C++: void (std::vector<OpenGLMesh::MeshDiagnostics> &, float) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    @overload
    def draw(self, shader: Shader) -> None:
        """
        /**
           * @brief Draw all instances using instanced rendering.
           *
           * No camera information, so projected LOD, mesh LOD selection and frustum
           * culling are not applied. Sets the batch uniforms (`shadingType`, `color1`,
           * `color2`, wind parameters) and uploads dirty instance data; the shader must
           * already be in use and a GL context current. Draws nothing while automatic
           * LODs are still being prepared.
           * @param shader Shader to use for drawing.
           */
        C++: void (Shader &)
        """

    @overload
    def draw(self, shader: Shader, viewPos: numpy.ndarray, verticalFovDeg: float, viewportHeight: int) -> None:
        """
        /**
           * @brief Draw all instances using camera-aware projected LOD.
           * @param shader Instanced shader that is already in use.
           * @param viewPos Camera position in world coordinates.
           * @param verticalFovDeg Camera vertical field of view in degrees.
           * @param viewportHeight Viewport height in pixels (<= 0 disables projected LOD).
           */
        C++: void (Shader &, const glm::vec3 &, float, int)
        """

    @overload
    def draw(self, shader: Shader, cullingShader: Shader | None, viewProjection: numpy.ndarray, viewPos: numpy.ndarray, verticalFovDeg: float, viewportHeight: int) -> None:
        """
        /**
           * @brief Draw all instances with optional GPU culling and indirect draws.
           *
           * @param shader Material shader used for the final draw.
           * @param cullingShader Optional compute shader. Null falls back to the CPU path.
           * @param viewProjection Current projection * view matrix for frustum culling.
           * @param viewPos Camera position in world coordinates.
           * @param verticalFovDeg Camera vertical field of view in degrees.
           * @param viewportHeight Viewport height in pixels.
           */
        C++: void (Shader &, Shader *, const glm::mat4 &, const glm::vec3 &, float, int)
        """

    @overload
    def draw(self, shader: Shader, cullingShader: Shader | None, gpuHizOcclusion: InstancedVisuals.GpuHizOcclusionConfig | None, viewProjection: numpy.ndarray, viewPos: numpy.ndarray, verticalFovDeg: float, viewportHeight: int) -> None:
        """
        /**
           * @brief As above, with optional hierarchical-Z occlusion culling.
           * @param shader Material shader used for the final draw.
           * @param cullingShader Optional instance-culling compute shader.
           * @param gpuHizOcclusion Optional Hi-Z input; null disables occlusion culling.
           * @param viewProjection Current projection * view matrix.
           * @param viewPos Camera position in world coordinates.
           * @param verticalFovDeg Camera vertical field of view in degrees.
           * @param viewportHeight Viewport height in pixels.
           */
        C++: void (Shader &, Shader *, const GpuHizOcclusionConfig *, const glm::mat4 &, const glm::vec3 &, float, int)
        """

    def drawShadow(self, shader: Shader, cullingShader: Shader | None, lightViewProjection: numpy.ndarray, viewPos: numpy.ndarray, verticalFovDeg: float, viewportHeight: int) -> None:
        """
        /**
           * @brief Draw this instanced visual for a shadow map with shadow-only foliage LOD/culling.
           * @param shader Shadow depth shader that is already in use.
           * @param cullingShader Optional instance-culling compute shader.
           * @param lightViewProjection Light-space view-projection matrix.
           * @param viewPos Main camera position (drives LOD distances).
           * @param verticalFovDeg Main camera vertical field of view in degrees.
           * @param viewportHeight Main viewport height in pixels.
           */
        C++: void (Shader &, Shader *, const glm::mat4 &, const glm::vec3 &, float, int)
        """

    def drawLinearDepth(self, shader: Shader, cullingShader: Shader | None, viewProjection: numpy.ndarray, viewPos: numpy.ndarray, verticalFovDeg: float, viewportHeight: int) -> None:
        """
        /**
           * @brief Draw this instanced visual into a linear-depth target.
           *
           * Culls instances and selects mesh LODs for the camera as draw() does, except for foliage,
           * which keeps its source meshes: simplified foliage drops leaf pieces, and a foliage
           * impostor's card holds the plant only in its color texture.
           * @param shader Linear-depth shader that is already in use.
           * @param cullingShader Optional instance-culling compute shader.
           * @param viewProjection Camera projection * view matrix.
           * @param viewPos Camera position in world coordinates.
           * @param verticalFovDeg Camera vertical field of view in degrees.
           * @param viewportHeight Depth target height in pixels.
           */
        C++: void (Shader &, Shader *, const glm::mat4 &, const glm::vec3 &, float, int)
        """

    def usedGpuInstanceCullingForDiagnostics(self) -> bool:
        """
        /**
           * @name Draw-path diagnostics
           * Internal: results of the most recent draw, for tests and renderer diagnostics.
           */
          ///@{
          /// @return True if GPU instance culling was used.
        C++: bool () const
        """

    def usedGpuHizOcclusionForDiagnostics(self) -> bool:
        """
        /// @return True if Hi-Z occlusion culling was used.
        C++: bool () const
        """

    def lastGpuInstanceCullCandidateCountForDiagnostics(self) -> int:
        """
        /// @return Instances submitted to GPU culling.
        C++: size_t () const
        """

    def lastGpuInstanceCullSkipCodeForDiagnostics(self) -> int:
        """
        /// @return 0, or why GPU culling was skipped.
        C++: uint32_t () const
        """

    def lastGpuInstanceCullSkipReasonForDiagnostics(self) -> int | None:
        """
        /// @return Text for lastGpuInstanceCullSkipCodeForDiagnostics().
        C++: const char *() const
        """

    def reusedVisibleInstanceFrameCacheForDiagnostics(self) -> bool:
        """
        /// @return True if the visible-instance list was reused from a previous frame.
        C++: bool () const
        """

    def visibleInstanceFrameCacheFallbackForDiagnostics(self) -> bool:
        """
        /// @return True if the visible-instance cache could not be used.
        C++: bool () const
        """

    def usedPersistentInstanceSourceBufferForDiagnostics(self) -> bool:
        """
        /// @return True if the persistent GPU instance source buffer was used.
        C++: bool () const
        """

    def persistentInstanceSourceBufferReadyForDiagnostics(self) -> bool:
        """
        /// @return True once the persistent GPU instance source buffer has been initialized.
        C++: bool () const
        """

    def gpuSourceInstanceCapacityForDiagnostics(self) -> int:
        """
        /// @return Capacity of the GPU instance source buffer, in instances.
        C++: size_t () const
        """

    def gpuSourceInstanceFullUploadCountForDiagnostics(self) -> int:
        """
        /// @return Full uploads of the GPU instance source buffer.
        C++: size_t () const
        """

    def gpuSourceInstanceSubUploadCountForDiagnostics(self) -> int:
        """
        /// @return Partial uploads of the GPU instance source buffer.
        C++: size_t () const
        """

    def compactInstanceFullUploadCountForDiagnostics(self) -> int:
        """
        /// @return Full uploads of the compact instance buffer.
        C++: size_t () const
        """

    def compactInstanceSubUploadCountForDiagnostics(self) -> int:
        """
        /// @return Partial uploads of the compact instance buffer.
        C++: size_t () const
        """

    def usedMultiDrawIndirectForDiagnostics(self) -> bool:
        """
        /// @return True if multi-draw-indirect was used.
        C++: bool () const
        """

    def lastMultiDrawIndirectCommandCountForDiagnostics(self) -> int:
        """
        /// @return Indirect commands issued by the last multi-draw.
        C++: size_t () const
        """

    def multiDrawIndirectCommandUploadCountForDiagnostics(self) -> int:
        """
        /// @return Uploads of the indirect command buffer.
        C++: size_t () const
        """

    def lastMultiDrawIndirectFallbackCodeForDiagnostics(self) -> int:
        """
        /// @return 0, or why multi-draw-indirect was not used.
        C++: uint32_t () const
        """

    def lastMultiDrawIndirectFallbackReasonForDiagnostics(self) -> int | None:
        """
        /// @return Text for lastMultiDrawIndirectFallbackCodeForDiagnostics().
        C++: const char *() const
        """

    def instanceDataGeneration(self) -> int:
        """
        ///@}
          /// Internal: @return Counter bumped whenever instance data changes (cache key).
        C++: uint64_t () const
        """

    def occlusionCandidatesPossible(self) -> bool:
        """
        /// Internal: whether addOcclusionCandidates() can add this batch: its draws take the
          /// bounds-culled path with enough geometry, before any frustum test.
        C++: bool () const
        """

    def addInstances(self, instances: Sequence[InstancedVisuals.InstanceSpec]) -> None: ...

class IrradianceVolume:
    def __init__(self) -> None: ...

    @property
    def center(self) -> numpy.ndarray: ...

    @center.setter
    def center(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def halfExtents(self) -> numpy.ndarray: ...

    @halfExtents.setter
    def halfExtents(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def orientation(self) -> numpy.ndarray: ...

    @orientation.setter
    def orientation(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def color(self) -> numpy.ndarray: ...

    @color.setter
    def color(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def strength(self) -> float: ...

    @strength.setter
    def strength(self, arg: float, /) -> None: ...

    @property
    def edgeFade(self) -> float: ...

    @edgeFade.setter
    def edgeFade(self, arg: float, /) -> None: ...

    @property
    def normalBias(self) -> float: ...

    @normalBias.setter
    def normalBias(self, arg: float, /) -> None: ...

class Light:
    def __init__(self, viewer: RayraiWindow) -> None: ...

    @property
    def type(self) -> LightType: ...

    @type.setter
    def type(self, arg: LightType, /) -> None: ...

    @property
    def ambient(self) -> numpy.ndarray: ...

    @ambient.setter
    def ambient(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def diffuse(self) -> numpy.ndarray: ...

    @diffuse.setter
    def diffuse(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def specular(self) -> numpy.ndarray: ...

    @specular.setter
    def specular(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def direction(self) -> numpy.ndarray: ...

    @direction.setter
    def direction(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def constant(self) -> float: ...

    @constant.setter
    def constant(self, arg: float, /) -> None: ...

    @property
    def linear(self) -> float: ...

    @linear.setter
    def linear(self, arg: float, /) -> None: ...

    @property
    def quadratic(self) -> float: ...

    @quadratic.setter
    def quadratic(self, arg: float, /) -> None: ...

    @property
    def spotInnerCos(self) -> float: ...

    @spotInnerCos.setter
    def spotInnerCos(self, arg: float, /) -> None: ...

    @property
    def spotOuterCos(self) -> float: ...

    @spotOuterCos.setter
    def spotOuterCos(self, arg: float, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def areaSize(self) -> numpy.ndarray: ...

    @areaSize.setter
    def areaSize(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def areaRight(self) -> numpy.ndarray: ...

    @areaRight.setter
    def areaRight(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def areaUp(self) -> numpy.ndarray: ...

    @areaUp.setter
    def areaUp(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def projectorMap(self) -> int: ...

    @projectorMap.setter
    def projectorMap(self, arg: int, /) -> None: ...

    @property
    def projectorStrength(self) -> float: ...

    @projectorStrength.setter
    def projectorStrength(self, arg: float, /) -> None: ...

    @property
    def projectorUvScaleOffset(self) -> numpy.ndarray: ...

    @projectorUvScaleOffset.setter
    def projectorUvScaleOffset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def negative(self) -> bool: ...

    @negative.setter
    def negative(self, arg: bool, /) -> None: ...

    @property
    def temperatureEnabled(self) -> bool: ...

    @temperatureEnabled.setter
    def temperatureEnabled(self, arg: bool, /) -> None: ...

    @property
    def temperatureKelvin(self) -> float: ...

    @temperatureKelvin.setter
    def temperatureKelvin(self, arg: float, /) -> None: ...

    @property
    def distanceFadeEnabled(self) -> bool: ...

    @distanceFadeEnabled.setter
    def distanceFadeEnabled(self, arg: bool, /) -> None: ...

    @property
    def distanceFadeBegin(self) -> float: ...

    @distanceFadeBegin.setter
    def distanceFadeBegin(self, arg: float, /) -> None: ...

    @property
    def distanceFadeShadow(self) -> float: ...

    @distanceFadeShadow.setter
    def distanceFadeShadow(self, arg: float, /) -> None: ...

    @property
    def distanceFadeLength(self) -> float: ...

    @distanceFadeLength.setter
    def distanceFadeLength(self, arg: float, /) -> None: ...

    def setAsDirectional(self, dir: numpy.ndarray, diffColor: numpy.ndarray = ...) -> None:
        """
        /**
           * @brief Configure as a directional light.
           * @param dir Direction pointing from the light toward the scene (normalized here).
           * @param diffColor Diffuse color (linear RGB).
           */
        C++: void (const glm::vec3 &, const glm::vec3 &)
        """

    def setAsPoint(self, pos: numpy.ndarray, diffColor: numpy.ndarray = ...) -> None:
        """
        /**
           * @brief Configure as a point light.
           * @param pos Light position in world coordinates.
           * @param diffColor Diffuse color (linear RGB).
           */
        C++: void (const glm::vec3 &, const glm::vec3 &)
        """

    def setSpotAngles(self, innerDeg: float, outerDeg: float) -> None:
        """
        /**
           * @brief Set the spotlight cone half-angles in degrees.
           *
           * Convenience wrapper around the raw cosine fields. The inner cone is the
           * fully-lit region; intensity falls off smoothly between `innerDeg` and
           * `outerDeg`. Pass any values; they are clamped so that
           * `innerDeg <= outerDeg`.
           *
           * @param innerDeg Half-angle in degrees where falloff begins.
           * @param outerDeg Half-angle in degrees where light reaches zero.
           */
        C++: void (float, float)
        """

    def setRange(self, rangeMeters: float) -> None:
        """
        /**
           * @brief Set the attenuation of a point, spot or area light from its range in metres.
           *
           * Sets the inverse-quadratic coefficients 1, 4.5/range and 75/range^2: full intensity at the
           * source, about 1% (1/80.5) at the range, and still decreasing beyond it (it never reaches zero,
           * and the range does not limit which objects the light shades). Use this instead of writing
           * `constant`/`linear`/`quadratic` by hand. #radius, the source size, is not changed.
           *
           * @param rangeMeters Distance at which the light contribution falls to about 1%.
           */
        C++: void (float)
        """

    def ensureShadowFramebuffer(self) -> bool:
        """
        /**
           * @brief Create or resize the shadow framebuffer and 2D depth texture.
           *
           * Requires a current OpenGL context. Resources created in a different
           * context are discarded and recreated.
           * @return Always true.
           * @throws std::runtime_error (via the framebuffer completeness check) if the
           *         framebuffer is incomplete; resources are discarded in that case.
           */
        C++: bool ()
        """

    def shadowPrePass(self) -> bool:
        """
        /**
           * @brief Prepare for 2D shadow depth rendering.
           *
           * Ensures the framebuffer, updates #lightSpaceMatrix_, binds the FBO, sets
           * the viewport, clears depth, enables polygon offset and disables face
           * culling (restored by shadowPostPass()). Requires a current GL context.
           * @return Always true; failures throw as in ensureShadowFramebuffer().
           */
        C++: bool ()
        """

    def shadowPrePassInto(self, framebuffer: int, clearDepth: bool) -> bool:
        """
        /**
           * @brief Prepare 2D shadow depth rendering into another framebuffer.
           *
           * Same as shadowPrePass() but binds @p framebuffer, whose depth attachment must match the
           * allocated shadow map size, and clears depth only when @p clearDepth is set. Call
           * ensureShadowFramebuffer() first.
           * @param framebuffer Framebuffer to render into.
           * @param clearDepth Clear the depth attachment before rendering.
           * @return Always true.
           */
        C++: bool (unsigned int, bool)
        """

    def updateShadowMatrix(self, snapResolution: int) -> None:
        """
        /**
           * @brief Recompute #lightSpaceMatrix_ (used by standalone shadow maps and atlas tiles).
           *
           * Directional lights use an orthographic box (see setShadowOrtho()) around
           * the shadow center; spot lights a perspective frustum fitted to the outer
           * cone; area lights a 150 degree frustum; point lights a 90 degree frustum.
           * @param snapResolution Shadow-map resolution in texels used to snap
           *        directional shadows to the texel grid (reduces shimmering).
           */
        C++: void (int)
        """

    def pointShadowPrePassFace(self, face: int) -> bool:
        """
        /**
           * @brief Prepare one face of a point-light cubemap shadow render.
           *
           * Updates #lightSpaceMatrix_ and the face's entry in
           * @c pointLightSpaceMatrices_, then binds that cubemap face for depth
           * rendering. Requires a current GL context.
           * @param face Cubemap face index in OpenGL order [0, 5] (clamped).
           * @return Always true; failures throw std::runtime_error.
           */
        C++: bool (int)
        """

    def shadowPostPass(self) -> None:
        """
        /**
           * @brief Restore the caller's face-cull state, disable polygon offset and
           *        bind the default framebuffer after shadow rendering.
           */
        C++: void ()
        """

    def setShadowResolution(self, res: int) -> None:
        """
        /**
           * @brief Set shadow-map resolution.
           *
           * Releases existing shadow resources when the value changes; they are
           * recreated at the new size on the next pre-pass.
           * @param res Shadow map size in pixels (min 64; also capped at the driver's
           *        maximum texture size on allocation).
           */
        C++: void (int)
        """

    def setShadowsEnabled(self, e: bool) -> None:
        """
        /**
           * @brief Enable or disable shadow sampling.
           * @param e True to enable shadows.
           */
        C++: void (bool)
        """

    def setShadowParams(self, bias: float, strength: float, pcfRadius: float) -> None:
        """
        /**
           * @brief Set shadow bias and filtering parameters.
           * @param bias Depth bias to reduce acne.
           * @param strength Shadow darkening strength.
           * @param pcfRadius PCF filter radius in shadow-map texels.
           */
        C++: void (float, float, float)
        """

    def setShadowOrtho(self, halfSize: float, zNear: float, zFar: float) -> None:
        """
        /**
           * @brief Configure the shadow projection volume.
           * @param halfSize Half-size of the orthographic box (directional lights only), in metres.
           * @param zNear Near plane in metres (all light types).
           * @param zFar Far plane in metres (all light types).
           */
        C++: void (float, float, float)
        """

    def setShadowCenter(self, center: numpy.ndarray) -> None:
        """
        /**
           * @brief Set the center point used for directional shadow projection.
           * @param center World-space center.
           */
        C++: void (const glm::vec3 &)
        """

    def setShadowPosition(self, pos: numpy.ndarray, enable: bool = True) -> None:
        """
        /**
           * @brief Override the light position used for directional shadow projection.
           * @param pos World-space light position.
           * @param enable True to use the custom position, false to revert to auto position.
           */
        C++: void (const glm::vec3 &, bool)
        """

class LightType(enum.Enum):
    DIRECTIONAL = 0

    POINT = 1

    SPOT = 2

    AREA = 3

class LocalFogVolume:
    def __init__(self) -> None: ...

    @property
    def center(self) -> numpy.ndarray: ...

    @center.setter
    def center(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def color(self) -> numpy.ndarray: ...

    @color.setter
    def color(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def density(self) -> float: ...

    @density.setter
    def density(self, arg: float, /) -> None: ...

    @property
    def edgeFade(self) -> float: ...

    @edgeFade.setter
    def edgeFade(self, arg: float, /) -> None: ...

    @property
    def noiseScale(self) -> float: ...

    @noiseScale.setter
    def noiseScale(self, arg: float, /) -> None: ...

    @property
    def noiseStrength(self) -> float: ...

    @noiseStrength.setter
    def noiseStrength(self, arg: float, /) -> None: ...

class Material:
    def __init__(self, name: str = 'DefaultMaterial') -> None: ...

    class AlphaAntiAliasing(enum.Enum):
        Off = 0

        AlphaToCoverage = 1

        AlphaToCoverageAndToOne = 2

    class AlphaMode(enum.Enum):
        Opaque = 0

        Mask = 1

        Hash = 2

        Blend = 3

    class AlphaSettings:
        def __init__(self) -> None: ...

        @property
        def mode(self) -> Material.AlphaMode: ...

        @mode.setter
        def mode(self, arg: Material.AlphaMode, /) -> None: ...

        @property
        def blend(self) -> Material.BlendMode: ...

        @blend.setter
        def blend(self, arg: Material.BlendMode, /) -> None: ...

        @property
        def antiAliasing(self) -> Material.AlphaAntiAliasing: ...

        @antiAliasing.setter
        def antiAliasing(self, arg: Material.AlphaAntiAliasing, /) -> None: ...

    class BillboardMode(enum.Enum):
        Disabled = 0

        Enabled = 1

        FixedZ = 2

        Particles = 3

    class BlendMode(enum.Enum):
        Mix = 0

        Add = 1

        Subtract = 2

        Multiply = 3

        PremultipliedAlpha = 4

    class CullMode(enum.Enum):
        Back = 0

        Front = 1

        Disabled = 2

    class DepthState(enum.Enum):
        Inherit = 0

        Enabled = 1

        Disabled = 2

        Inverted = 3

    class DetailBlendMode(enum.Enum):
        SoftMultiply = 0

        Mix = 1

        Add = 2

        Subtract = 3

        Multiply = 4

    class DiffuseMode(enum.Enum):
        Burley = 0

        Lambert = 1

        LambertWrap = 2

        Toon = 3

    class DistanceFadeMode(enum.Enum):
        Disabled = 0

        PixelAlpha = 1

        PixelDither = 2

        ObjectDither = 3

    class EmissionOperator(enum.Enum):
        Add = 0

        Multiply = 1

    class FoliageSettings:
        def __init__(self) -> None: ...

        @property
        def type(self) -> Material.FoliageType: ...

        @type.setter
        def type(self, arg: Material.FoliageType, /) -> None: ...

        @property
        def windStrength(self) -> float: ...

        @windStrength.setter
        def windStrength(self, arg: float, /) -> None: ...

        @property
        def rootHeight(self) -> float: ...

        @rootHeight.setter
        def rootHeight(self, arg: float, /) -> None: ...

        @property
        def tipHeight(self) -> float: ...

        @tipHeight.setter
        def tipHeight(self, arg: float, /) -> None: ...

        @property
        def stiffness(self) -> float: ...

        @stiffness.setter
        def stiffness(self, arg: float, /) -> None: ...

        @property
        def flutterWeight(self) -> float: ...

        @flutterWeight.setter
        def flutterWeight(self, arg: float, /) -> None: ...

        @property
        def twoSidedLighting(self) -> bool: ...

        @twoSidedLighting.setter
        def twoSidedLighting(self, arg: bool, /) -> None: ...

        @property
        def transmissionStrength(self) -> float: ...

        @transmissionStrength.setter
        def transmissionStrength(self, arg: float, /) -> None: ...

        @property
        def transmissionColor(self) -> numpy.ndarray: ...

        @transmissionColor.setter
        def transmissionColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def weatherResponse(self) -> bool: ...

        @weatherResponse.setter
        def weatherResponse(self, arg: bool, /) -> None: ...

        @property
        def wetResponse(self) -> float: ...

        @wetResponse.setter
        def wetResponse(self, arg: float, /) -> None: ...

        @property
        def snowResponse(self) -> float: ...

        @snowResponse.setter
        def snowResponse(self, arg: float, /) -> None: ...

    class FoliageType(enum.Enum):
        Grass = 1

        LeafCard = 2

        Bush = 3

        Branch = 4

        TreeTrunk = 5

        Crop = 6

        Vine = 7

    class NormalMapConvention(enum.Enum):
        OpenGL = 0

        DirectX = 1

    class ParallaxSettings:
        def __init__(self) -> None: ...

        @property
        def scale(self) -> float: ...

        @scale.setter
        def scale(self, arg: float, /) -> None: ...

        @property
        def deepParallax(self) -> bool: ...

        @deepParallax.setter
        def deepParallax(self, arg: bool, /) -> None: ...

        @property
        def minLayers(self) -> int: ...

        @minLayers.setter
        def minLayers(self, arg: int, /) -> None: ...

        @property
        def maxLayers(self) -> int: ...

        @maxLayers.setter
        def maxLayers(self, arg: int, /) -> None: ...

        @property
        def flip(self) -> numpy.ndarray: ...

        @flip.setter
        def flip(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def invert(self) -> bool: ...

        @invert.setter
        def invert(self, arg: bool, /) -> None: ...

    class ShadingModel(enum.Enum):
        Standard = 0

        Lambert = 1

        Toon = 2

        Unlit = 3

    class SpecularMode(enum.Enum):
        SchlickGgx = 0

        Toon = 1

        Disabled = 2

    class StencilCompare(enum.Enum):
        Always = 0

        Less = 1

        Equal = 2

        LessOrEqual = 3

        Greater = 4

        NotEqual = 5

        GreaterOrEqual = 6

    class StencilEffectMode(enum.Enum):
        Disabled = 0

        Outline = 1

        Xray = 2

        Custom = 3

    class TextureBindingInfo:
        def __init__(self) -> None: ...

        @property
        def slot(self) -> str: ...

        @property
        def samplerUniform(self) -> str: ...

        @property
        def presenceUniform(self) -> str: ...

        @property
        def target(self) -> str: ...

        @property
        def textureUnit(self) -> int: ...

        @textureUnit.setter
        def textureUnit(self, arg: int, /) -> None: ...

    class TextureChannel(enum.Enum):
        Red = 0

        Green = 1

        Blue = 2

        Alpha = 3

        Gray = 4

    class TextureFilter(enum.Enum):
        Inherit = 0

        Nearest = 1

        Linear = 2

        NearestWithMipmaps = 3

        LinearWithMipmaps = 4

        NearestWithMipmapsAnisotropic = 5

        LinearWithMipmapsAnisotropic = 6

    class TextureRepeatMode(enum.Enum):
        Inherit = 0

        Repeat = 1

        Disabled = 2

        Mirror = 3

    class TextureSlot(enum.Enum):
        Albedo = 0

        Normal = 1

        BentNormal = 2

        Metallic = 3

        Roughness = 4

        MetallicRoughness = 5

        Ao = 6

        Emissive = 7

        Clearcoat = 8

        ClearcoatRoughness = 9

        ClearcoatNormal = 10

        SheenColor = 11

        SheenRoughness = 12

        Transmission = 13

        Refraction = 14

        Thickness = 15

        Subsurface = 16

        SubsurfaceTransmittance = 17

        Backlight = 18

        Anisotropy = 19

        WeatherMask = 20

        Rim = 21

        Height = 22

        DetailMask = 23

        DetailAlbedo = 24

        DetailNormal = 25

        Lightmap = 26

        TextureBlend = 27

        Count = 28

    class TextureSlotInfo:
        def __init__(self) -> None: ...

        @property
        def name(self) -> str: ...

        @property
        def isSrgb(self) -> bool: ...

        @isSrgb.setter
        def isSrgb(self, arg: bool, /) -> None: ...

        @property
        def hasUvTransform(self) -> bool: ...

        @hasUvTransform.setter
        def hasUvTransform(self, arg: bool, /) -> None: ...

    class Type(enum.Enum):
        SIMPLE_COLOR = 0

        TEXTURED = 1

        PBR = 2

    class UvLayer(enum.Enum):
        Uv1 = 0

        Uv2 = 1

    class UvTransform:
        def __init__(self) -> None: ...

        @property
        def scaleOffset(self) -> numpy.ndarray: ...

        @scaleOffset.setter
        def scaleOffset(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def rotation(self) -> float: ...

        @rotation.setter
        def rotation(self, arg: float, /) -> None: ...

        @property
        def authored(self) -> bool: ...

        @authored.setter
        def authored(self, arg: bool, /) -> None: ...

    class VertexDisplacementSettings:
        def __init__(self) -> None: ...

        @property
        def scale(self) -> float: ...

        @scale.setter
        def scale(self, arg: float, /) -> None: ...

        @property
        def midlevel(self) -> float: ...

        @midlevel.setter
        def midlevel(self, arg: float, /) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def type(self) -> Material.Type: ...

    @type.setter
    def type(self, arg: Material.Type, /) -> None: ...

    @property
    def forceSimpleShading(self) -> bool: ...

    @forceSimpleShading.setter
    def forceSimpleShading(self, arg: bool, /) -> None: ...

    @property
    def baseColorFactor(self) -> numpy.ndarray: ...

    @baseColorFactor.setter
    def baseColorFactor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def useVertexColorAsAlbedo(self) -> bool: ...

    @useVertexColorAsAlbedo.setter
    def useVertexColorAsAlbedo(self, arg: bool, /) -> None: ...

    @property
    def vertexColorIsSrgb(self) -> bool: ...

    @vertexColorIsSrgb.setter
    def vertexColorIsSrgb(self, arg: bool, /) -> None: ...

    @property
    def textureBlendFactor(self) -> float: ...

    @textureBlendFactor.setter
    def textureBlendFactor(self, arg: float, /) -> None: ...

    @property
    def metallicFactor(self) -> float: ...

    @metallicFactor.setter
    def metallicFactor(self, arg: float, /) -> None: ...

    @property
    def roughnessFactor(self) -> float: ...

    @roughnessFactor.setter
    def roughnessFactor(self, arg: float, /) -> None: ...

    @property
    def ao(self) -> float: ...

    @ao.setter
    def ao(self, arg: float, /) -> None: ...

    @property
    def emissiveFactor(self) -> numpy.ndarray: ...

    @emissiveFactor.setter
    def emissiveFactor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def iblStrength(self) -> float: ...

    @iblStrength.setter
    def iblStrength(self, arg: float, /) -> None: ...

    @property
    def environmentMapStrength(self) -> float: ...

    @environmentMapStrength.setter
    def environmentMapStrength(self, arg: float, /) -> None: ...

    @property
    def environmentBoxProjection(self) -> bool: ...

    @environmentBoxProjection.setter
    def environmentBoxProjection(self, arg: bool, /) -> None: ...

    @property
    def environmentProbePosition(self) -> numpy.ndarray: ...

    @environmentProbePosition.setter
    def environmentProbePosition(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def environmentBoxMin(self) -> numpy.ndarray: ...

    @environmentBoxMin.setter
    def environmentBoxMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def environmentBoxMax(self) -> numpy.ndarray: ...

    @environmentBoxMax.setter
    def environmentBoxMax(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def normalStrength(self) -> float: ...

    @normalStrength.setter
    def normalStrength(self, arg: float, /) -> None: ...

    @property
    def normalMapYScale(self) -> float: ...

    @normalMapYScale.setter
    def normalMapYScale(self, arg: float, /) -> None: ...

    @property
    def bentNormalStrength(self) -> float: ...

    @bentNormalStrength.setter
    def bentNormalStrength(self, arg: float, /) -> None: ...

    @property
    def normalMapConvention(self) -> Material.NormalMapConvention: ...

    @normalMapConvention.setter
    def normalMapConvention(self, arg: Material.NormalMapConvention, /) -> None: ...

    @property
    def alphaMode(self) -> Material.AlphaMode: ...

    @alphaMode.setter
    def alphaMode(self, arg: Material.AlphaMode, /) -> None: ...

    @property
    def blendMode(self) -> Material.BlendMode: ...

    @blendMode.setter
    def blendMode(self, arg: Material.BlendMode, /) -> None: ...

    @property
    def renderPriority(self) -> int: ...

    @renderPriority.setter
    def renderPriority(self, arg: int, /) -> None: ...

    @property
    def nextPass(self) -> Material: ...

    @nextPass.setter
    def nextPass(self, arg: Material, /) -> None: ...

    @property
    def alphaCutoff(self) -> float: ...

    @alphaCutoff.setter
    def alphaCutoff(self, arg: float, /) -> None: ...

    @property
    def alphaHashScale(self) -> float: ...

    @alphaHashScale.setter
    def alphaHashScale(self, arg: float, /) -> None: ...

    @property
    def alphaAntialiasing(self) -> Material.AlphaAntiAliasing: ...

    @alphaAntialiasing.setter
    def alphaAntialiasing(self, arg: Material.AlphaAntiAliasing, /) -> None: ...

    @property
    def alphaAntialiasingEdge(self) -> float: ...

    @alphaAntialiasingEdge.setter
    def alphaAntialiasingEdge(self, arg: float, /) -> None: ...

    @property
    def distanceFadeMode(self) -> Material.DistanceFadeMode: ...

    @distanceFadeMode.setter
    def distanceFadeMode(self, arg: Material.DistanceFadeMode, /) -> None: ...

    @property
    def distanceFadeMin(self) -> float: ...

    @distanceFadeMin.setter
    def distanceFadeMin(self, arg: float, /) -> None: ...

    @property
    def distanceFadeMax(self) -> float: ...

    @distanceFadeMax.setter
    def distanceFadeMax(self, arg: float, /) -> None: ...

    @property
    def proximityFadeEnabled(self) -> bool: ...

    @proximityFadeEnabled.setter
    def proximityFadeEnabled(self, arg: bool, /) -> None: ...

    @property
    def proximityFadeDistance(self) -> float: ...

    @proximityFadeDistance.setter
    def proximityFadeDistance(self, arg: float, /) -> None: ...

    @property
    def aoStrength(self) -> float: ...

    @aoStrength.setter
    def aoStrength(self, arg: float, /) -> None: ...

    @property
    def aoLightAffect(self) -> float: ...

    @aoLightAffect.setter
    def aoLightAffect(self, arg: float, /) -> None: ...

    @property
    def emissiveStrength(self) -> float: ...

    @emissiveStrength.setter
    def emissiveStrength(self, arg: float, /) -> None: ...

    @property
    def emissionOperator(self) -> Material.EmissionOperator: ...

    @emissionOperator.setter
    def emissionOperator(self, arg: Material.EmissionOperator, /) -> None: ...

    @property
    def specularFactor(self) -> float: ...

    @specularFactor.setter
    def specularFactor(self, arg: float, /) -> None: ...

    @property
    def specularColorFactor(self) -> numpy.ndarray: ...

    @specularColorFactor.setter
    def specularColorFactor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def diffuseMode(self) -> Material.DiffuseMode: ...

    @diffuseMode.setter
    def diffuseMode(self, arg: Material.DiffuseMode, /) -> None: ...

    @property
    def specularMode(self) -> Material.SpecularMode: ...

    @specularMode.setter
    def specularMode(self, arg: Material.SpecularMode, /) -> None: ...

    @property
    def ior(self) -> float: ...

    @ior.setter
    def ior(self, arg: float, /) -> None: ...

    @property
    def transmissionFactor(self) -> float: ...

    @transmissionFactor.setter
    def transmissionFactor(self, arg: float, /) -> None: ...

    @property
    def refractionFactor(self) -> float: ...

    @refractionFactor.setter
    def refractionFactor(self, arg: float, /) -> None: ...

    @property
    def volumeThicknessFactor(self) -> float: ...

    @volumeThicknessFactor.setter
    def volumeThicknessFactor(self, arg: float, /) -> None: ...

    @property
    def dielectricPriority(self) -> int: ...

    @dielectricPriority.setter
    def dielectricPriority(self, arg: int, /) -> None: ...

    @property
    def dielectricVolumeId(self) -> int: ...

    @dielectricVolumeId.setter
    def dielectricVolumeId(self, arg: int, /) -> None: ...

    @property
    def volumeAttenuationDistance(self) -> float: ...

    @volumeAttenuationDistance.setter
    def volumeAttenuationDistance(self, arg: float, /) -> None: ...

    @property
    def volumeAttenuationColor(self) -> numpy.ndarray: ...

    @volumeAttenuationColor.setter
    def volumeAttenuationColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def subsurfaceScatteringStrength(self) -> float: ...

    @subsurfaceScatteringStrength.setter
    def subsurfaceScatteringStrength(self, arg: float, /) -> None: ...

    @property
    def subsurfaceTransmittanceColor(self) -> numpy.ndarray: ...

    @subsurfaceTransmittanceColor.setter
    def subsurfaceTransmittanceColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def subsurfaceTransmittanceDepth(self) -> float: ...

    @subsurfaceTransmittanceDepth.setter
    def subsurfaceTransmittanceDepth(self, arg: float, /) -> None: ...

    @property
    def subsurfaceTransmittanceBoost(self) -> float: ...

    @subsurfaceTransmittanceBoost.setter
    def subsurfaceTransmittanceBoost(self, arg: float, /) -> None: ...

    @property
    def subsurfaceSkinMode(self) -> bool: ...

    @subsurfaceSkinMode.setter
    def subsurfaceSkinMode(self, arg: bool, /) -> None: ...

    @property
    def backlightColor(self) -> numpy.ndarray: ...

    @backlightColor.setter
    def backlightColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def metallicTextureChannel(self) -> Material.TextureChannel: ...

    @metallicTextureChannel.setter
    def metallicTextureChannel(self, arg: Material.TextureChannel, /) -> None: ...

    @property
    def roughnessTextureChannel(self) -> Material.TextureChannel: ...

    @roughnessTextureChannel.setter
    def roughnessTextureChannel(self, arg: Material.TextureChannel, /) -> None: ...

    @property
    def aoTextureChannel(self) -> Material.TextureChannel: ...

    @aoTextureChannel.setter
    def aoTextureChannel(self, arg: Material.TextureChannel, /) -> None: ...

    @property
    def refractionTextureChannel(self) -> Material.TextureChannel: ...

    @refractionTextureChannel.setter
    def refractionTextureChannel(self, arg: Material.TextureChannel, /) -> None: ...

    @property
    def uvScaleOffset(self) -> numpy.ndarray: ...

    @uvScaleOffset.setter
    def uvScaleOffset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def uvRotation(self) -> float: ...

    @uvRotation.setter
    def uvRotation(self, arg: float, /) -> None: ...

    @property
    def uv2ScaleOffset(self) -> numpy.ndarray: ...

    @uv2ScaleOffset.setter
    def uv2ScaleOffset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def uv2Rotation(self) -> float: ...

    @uv2Rotation.setter
    def uv2Rotation(self, arg: float, /) -> None: ...

    @property
    def textureRepeatMode(self) -> Material.TextureRepeatMode: ...

    @textureRepeatMode.setter
    def textureRepeatMode(self, arg: Material.TextureRepeatMode, /) -> None: ...

    @property
    def textureFilter(self) -> Material.TextureFilter: ...

    @textureFilter.setter
    def textureFilter(self, arg: Material.TextureFilter, /) -> None: ...

    @property
    def textureFilterAnisotropy(self) -> float: ...

    @textureFilterAnisotropy.setter
    def textureFilterAnisotropy(self, arg: float, /) -> None: ...

    @property
    def albedoTextureForceSrgb(self) -> bool: ...

    @albedoTextureForceSrgb.setter
    def albedoTextureForceSrgb(self, arg: bool, /) -> None: ...

    @property
    def albedoTextureMsdf(self) -> bool: ...

    @albedoTextureMsdf.setter
    def albedoTextureMsdf(self, arg: bool, /) -> None: ...

    @property
    def msdfPixelRange(self) -> float: ...

    @msdfPixelRange.setter
    def msdfPixelRange(self, arg: float, /) -> None: ...

    @property
    def msdfOutlineSize(self) -> float: ...

    @msdfOutlineSize.setter
    def msdfOutlineSize(self, arg: float, /) -> None: ...

    @property
    def billboardMode(self) -> Material.BillboardMode: ...

    @billboardMode.setter
    def billboardMode(self, arg: Material.BillboardMode, /) -> None: ...

    @property
    def billboardKeepScale(self) -> bool: ...

    @billboardKeepScale.setter
    def billboardKeepScale(self, arg: bool, /) -> None: ...

    @property
    def particlesAnimHFrames(self) -> int: ...

    @particlesAnimHFrames.setter
    def particlesAnimHFrames(self, arg: int, /) -> None: ...

    @property
    def particlesAnimVFrames(self) -> int: ...

    @particlesAnimVFrames.setter
    def particlesAnimVFrames(self, arg: int, /) -> None: ...

    @property
    def particlesAnimLoop(self) -> bool: ...

    @particlesAnimLoop.setter
    def particlesAnimLoop(self, arg: bool, /) -> None: ...

    @property
    def particlesAnimFrameProgress(self) -> float: ...

    @particlesAnimFrameProgress.setter
    def particlesAnimFrameProgress(self, arg: float, /) -> None: ...

    @property
    def particleTrailsEnabled(self) -> bool: ...

    @particleTrailsEnabled.setter
    def particleTrailsEnabled(self, arg: bool, /) -> None: ...

    @property
    def particleTrailLength(self) -> float: ...

    @particleTrailLength.setter
    def particleTrailLength(self, arg: float, /) -> None: ...

    @property
    def particleTrailTaper(self) -> float: ...

    @particleTrailTaper.setter
    def particleTrailTaper(self, arg: float, /) -> None: ...

    @property
    def particleTrailWorldDirection(self) -> numpy.ndarray: ...

    @particleTrailWorldDirection.setter
    def particleTrailWorldDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def fixedSize(self) -> bool: ...

    @fixedSize.setter
    def fixedSize(self, arg: bool, /) -> None: ...

    @property
    def fixedSizeScale(self) -> float: ...

    @fixedSizeScale.setter
    def fixedSizeScale(self, arg: float, /) -> None: ...

    @property
    def usePointSize(self) -> bool: ...

    @usePointSize.setter
    def usePointSize(self, arg: bool, /) -> None: ...

    @property
    def pointSize(self) -> float: ...

    @pointSize.setter
    def pointSize(self, arg: float, /) -> None: ...

    @property
    def useZClipScale(self) -> bool: ...

    @useZClipScale.setter
    def useZClipScale(self, arg: bool, /) -> None: ...

    @property
    def zClipScale(self) -> float: ...

    @zClipScale.setter
    def zClipScale(self, arg: float, /) -> None: ...

    @property
    def useFovOverride(self) -> bool: ...

    @useFovOverride.setter
    def useFovOverride(self, arg: bool, /) -> None: ...

    @property
    def fovOverrideDegrees(self) -> float: ...

    @fovOverrideDegrees.setter
    def fovOverrideDegrees(self, arg: float, /) -> None: ...

    @property
    def triplanarEnabled(self) -> bool: ...

    @triplanarEnabled.setter
    def triplanarEnabled(self, arg: bool, /) -> None: ...

    @property
    def triplanarWorldSpace(self) -> bool: ...

    @triplanarWorldSpace.setter
    def triplanarWorldSpace(self, arg: bool, /) -> None: ...

    @property
    def triplanarScale(self) -> numpy.ndarray: ...

    @triplanarScale.setter
    def triplanarScale(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def triplanarOffset(self) -> numpy.ndarray: ...

    @triplanarOffset.setter
    def triplanarOffset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def triplanarSharpness(self) -> float: ...

    @triplanarSharpness.setter
    def triplanarSharpness(self, arg: float, /) -> None: ...

    @property
    def uv2TriplanarEnabled(self) -> bool: ...

    @uv2TriplanarEnabled.setter
    def uv2TriplanarEnabled(self, arg: bool, /) -> None: ...

    @property
    def uv2TriplanarWorldSpace(self) -> bool: ...

    @uv2TriplanarWorldSpace.setter
    def uv2TriplanarWorldSpace(self, arg: bool, /) -> None: ...

    @property
    def uv2TriplanarScale(self) -> numpy.ndarray: ...

    @uv2TriplanarScale.setter
    def uv2TriplanarScale(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def uv2TriplanarOffset(self) -> numpy.ndarray: ...

    @uv2TriplanarOffset.setter
    def uv2TriplanarOffset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def uv2TriplanarSharpness(self) -> float: ...

    @uv2TriplanarSharpness.setter
    def uv2TriplanarSharpness(self, arg: float, /) -> None: ...

    @property
    def textureUvTransforms(self) -> list[Material.UvTransform]: ...

    @textureUvTransforms.setter
    def textureUvTransforms(self, arg: Sequence[Material.UvTransform], /) -> None: ...

    @property
    def clearcoatFactor(self) -> float: ...

    @clearcoatFactor.setter
    def clearcoatFactor(self, arg: float, /) -> None: ...

    @property
    def clearcoatRoughnessFactor(self) -> float: ...

    @clearcoatRoughnessFactor.setter
    def clearcoatRoughnessFactor(self, arg: float, /) -> None: ...

    @property
    def iridescenceFactor(self) -> float: ...

    @iridescenceFactor.setter
    def iridescenceFactor(self, arg: float, /) -> None: ...

    @property
    def iridescenceThickness(self) -> float: ...

    @iridescenceThickness.setter
    def iridescenceThickness(self, arg: float, /) -> None: ...

    @property
    def sheenFactor(self) -> float: ...

    @sheenFactor.setter
    def sheenFactor(self, arg: float, /) -> None: ...

    @property
    def sheenColorFactor(self) -> numpy.ndarray: ...

    @sheenColorFactor.setter
    def sheenColorFactor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def sheenRoughnessFactor(self) -> float: ...

    @sheenRoughnessFactor.setter
    def sheenRoughnessFactor(self, arg: float, /) -> None: ...

    @property
    def anisotropyFactor(self) -> float: ...

    @anisotropyFactor.setter
    def anisotropyFactor(self, arg: float, /) -> None: ...

    @property
    def anisotropyRotation(self) -> float: ...

    @anisotropyRotation.setter
    def anisotropyRotation(self, arg: float, /) -> None: ...

    @property
    def unlit(self) -> bool: ...

    @unlit.setter
    def unlit(self, arg: bool, /) -> None: ...

    @property
    def proceduralGroundChecker(self) -> bool: ...

    @proceduralGroundChecker.setter
    def proceduralGroundChecker(self, arg: bool, /) -> None: ...

    @property
    def usePlanarReflection(self) -> bool: ...

    @usePlanarReflection.setter
    def usePlanarReflection(self, arg: bool, /) -> None: ...

    @property
    def planarReflectionStrength(self) -> float: ...

    @planarReflectionStrength.setter
    def planarReflectionStrength(self, arg: float, /) -> None: ...

    @property
    def dontReceiveShadows(self) -> bool: ...

    @dontReceiveShadows.setter
    def dontReceiveShadows(self, arg: bool, /) -> None: ...

    @property
    def disableAmbientLight(self) -> bool: ...

    @disableAmbientLight.setter
    def disableAmbientLight(self, arg: bool, /) -> None: ...

    @property
    def disableFog(self) -> bool: ...

    @disableFog.setter
    def disableFog(self, arg: bool, /) -> None: ...

    @property
    def disableSpecularOcclusion(self) -> bool: ...

    @disableSpecularOcclusion.setter
    def disableSpecularOcclusion(self, arg: bool, /) -> None: ...

    @property
    def shadowToOpacity(self) -> bool: ...

    @shadowToOpacity.setter
    def shadowToOpacity(self, arg: bool, /) -> None: ...

    @property
    def doubleSided(self) -> bool: ...

    @doubleSided.setter
    def doubleSided(self, arg: bool, /) -> None: ...

    @property
    def cullMode(self) -> Material.CullMode: ...

    @cullMode.setter
    def cullMode(self, arg: Material.CullMode, /) -> None: ...

    @property
    def thinTwoSidedLighting(self) -> bool: ...

    @thinTwoSidedLighting.setter
    def thinTwoSidedLighting(self, arg: bool, /) -> None: ...

    @property
    def thinBackDiffuseScale(self) -> float: ...

    @thinBackDiffuseScale.setter
    def thinBackDiffuseScale(self, arg: float, /) -> None: ...

    @property
    def thinTransmissionStrength(self) -> float: ...

    @thinTransmissionStrength.setter
    def thinTransmissionStrength(self, arg: float, /) -> None: ...

    @property
    def thinTransmissionColor(self) -> numpy.ndarray: ...

    @thinTransmissionColor.setter
    def thinTransmissionColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def foliageType(self) -> Material.FoliageType: ...

    @foliageType.setter
    def foliageType(self, arg: Material.FoliageType, /) -> None: ...

    @property
    def foliageWindStrength(self) -> float: ...

    @foliageWindStrength.setter
    def foliageWindStrength(self, arg: float, /) -> None: ...

    @property
    def foliageRootHeight(self) -> float: ...

    @foliageRootHeight.setter
    def foliageRootHeight(self, arg: float, /) -> None: ...

    @property
    def foliageTipHeight(self) -> float: ...

    @foliageTipHeight.setter
    def foliageTipHeight(self, arg: float, /) -> None: ...

    @property
    def foliageStiffness(self) -> float: ...

    @foliageStiffness.setter
    def foliageStiffness(self, arg: float, /) -> None: ...

    @property
    def foliageFlutterWeight(self) -> float: ...

    @foliageFlutterWeight.setter
    def foliageFlutterWeight(self, arg: float, /) -> None: ...

    @property
    def foliageTwoSidedLighting(self) -> bool: ...

    @foliageTwoSidedLighting.setter
    def foliageTwoSidedLighting(self, arg: bool, /) -> None: ...

    @property
    def foliageDepthLodViewCount(self) -> int: ...

    @foliageDepthLodViewCount.setter
    def foliageDepthLodViewCount(self, arg: int, /) -> None: ...

    @property
    def foliageDepthLodBlendWidth(self) -> float: ...

    @foliageDepthLodBlendWidth.setter
    def foliageDepthLodBlendWidth(self, arg: float, /) -> None: ...

    @property
    def foliageTransmissionStrength(self) -> float: ...

    @foliageTransmissionStrength.setter
    def foliageTransmissionStrength(self, arg: float, /) -> None: ...

    @property
    def foliageTransmissionColor(self) -> numpy.ndarray: ...

    @foliageTransmissionColor.setter
    def foliageTransmissionColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def foliageWeatherResponse(self) -> bool: ...

    @foliageWeatherResponse.setter
    def foliageWeatherResponse(self, arg: bool, /) -> None: ...

    @property
    def foliageWetResponse(self) -> float: ...

    @foliageWetResponse.setter
    def foliageWetResponse(self, arg: float, /) -> None: ...

    @property
    def foliageSnowResponse(self) -> float: ...

    @foliageSnowResponse.setter
    def foliageSnowResponse(self, arg: float, /) -> None: ...

    @property
    def weatherAffectsWetness(self) -> bool: ...

    @weatherAffectsWetness.setter
    def weatherAffectsWetness(self, arg: bool, /) -> None: ...

    @property
    def weatherAffectsSnow(self) -> bool: ...

    @weatherAffectsSnow.setter
    def weatherAffectsSnow(self, arg: bool, /) -> None: ...

    @property
    def weatherPorosity(self) -> float: ...

    @weatherPorosity.setter
    def weatherPorosity(self, arg: float, /) -> None: ...

    @property
    def weatherPuddleEligible(self) -> bool: ...

    @weatherPuddleEligible.setter
    def weatherPuddleEligible(self, arg: bool, /) -> None: ...

    @property
    def weatherSnowEligible(self) -> bool: ...

    @weatherSnowEligible.setter
    def weatherSnowEligible(self, arg: bool, /) -> None: ...

    @property
    def rimFactor(self) -> float: ...

    @rimFactor.setter
    def rimFactor(self, arg: float, /) -> None: ...

    @property
    def rimTint(self) -> float: ...

    @rimTint.setter
    def rimTint(self, arg: float, /) -> None: ...

    @property
    def rimPower(self) -> float: ...

    @rimPower.setter
    def rimPower(self, arg: float, /) -> None: ...

    @property
    def rimColor(self) -> numpy.ndarray: ...

    @rimColor.setter
    def rimColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def growEnabled(self) -> bool: ...

    @growEnabled.setter
    def growEnabled(self, arg: bool, /) -> None: ...

    @property
    def growAmount(self) -> float: ...

    @growAmount.setter
    def growAmount(self, arg: float, /) -> None: ...

    @property
    def heightmapScale(self) -> float: ...

    @heightmapScale.setter
    def heightmapScale(self, arg: float, /) -> None: ...

    @property
    def heightmapVertexDisplacementScale(self) -> float: ...

    @heightmapVertexDisplacementScale.setter
    def heightmapVertexDisplacementScale(self, arg: float, /) -> None: ...

    @property
    def heightmapVertexDisplacementMidlevel(self) -> float: ...

    @heightmapVertexDisplacementMidlevel.setter
    def heightmapVertexDisplacementMidlevel(self, arg: float, /) -> None: ...

    @property
    def heightmapDeepParallax(self) -> bool: ...

    @heightmapDeepParallax.setter
    def heightmapDeepParallax(self, arg: bool, /) -> None: ...

    @property
    def heightmapMinLayers(self) -> int: ...

    @heightmapMinLayers.setter
    def heightmapMinLayers(self, arg: int, /) -> None: ...

    @property
    def heightmapMaxLayers(self) -> int: ...

    @heightmapMaxLayers.setter
    def heightmapMaxLayers(self, arg: int, /) -> None: ...

    @property
    def heightmapFlip(self) -> numpy.ndarray: ...

    @heightmapFlip.setter
    def heightmapFlip(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def heightmapInvert(self) -> bool: ...

    @heightmapInvert.setter
    def heightmapInvert(self, arg: bool, /) -> None: ...

    @property
    def detailAlbedoBlend(self) -> float: ...

    @detailAlbedoBlend.setter
    def detailAlbedoBlend(self, arg: float, /) -> None: ...

    @property
    def detailUvLayer(self) -> Material.UvLayer: ...

    @detailUvLayer.setter
    def detailUvLayer(self, arg: Material.UvLayer, /) -> None: ...

    @property
    def detailAlbedoBlendMode(self) -> Material.DetailBlendMode: ...

    @detailAlbedoBlendMode.setter
    def detailAlbedoBlendMode(self, arg: Material.DetailBlendMode, /) -> None: ...

    @property
    def detailNormalStrength(self) -> float: ...

    @detailNormalStrength.setter
    def detailNormalStrength(self, arg: float, /) -> None: ...

    @property
    def aoUsesUv2(self) -> bool: ...

    @aoUsesUv2.setter
    def aoUsesUv2(self, arg: bool, /) -> None: ...

    @property
    def emissiveUsesUv2(self) -> bool: ...

    @emissiveUsesUv2.setter
    def emissiveUsesUv2(self, arg: bool, /) -> None: ...

    @property
    def lightmapUsesUv2(self) -> bool: ...

    @lightmapUsesUv2.setter
    def lightmapUsesUv2(self, arg: bool, /) -> None: ...

    @property
    def lightmapStrength(self) -> float: ...

    @lightmapStrength.setter
    def lightmapStrength(self, arg: float, /) -> None: ...

    @property
    def lightmapTint(self) -> numpy.ndarray: ...

    @lightmapTint.setter
    def lightmapTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def decalDepthBias(self) -> bool: ...

    @decalDepthBias.setter
    def decalDepthBias(self, arg: bool, /) -> None: ...

    @property
    def tinySurfaceDetail(self) -> bool: ...

    @tinySurfaceDetail.setter
    def tinySurfaceDetail(self, arg: bool, /) -> None: ...

    @property
    def depthTest(self) -> Material.DepthState: ...

    @depthTest.setter
    def depthTest(self, arg: Material.DepthState, /) -> None: ...

    @property
    def depthWrite(self) -> Material.DepthState: ...

    @depthWrite.setter
    def depthWrite(self, arg: Material.DepthState, /) -> None: ...

    @property
    def stencilReference(self) -> int: ...

    @stencilReference.setter
    def stencilReference(self, arg: int, /) -> None: ...

    @property
    def stencilRead(self) -> bool: ...

    @stencilRead.setter
    def stencilRead(self, arg: bool, /) -> None: ...

    @property
    def stencilWrite(self) -> bool: ...

    @stencilWrite.setter
    def stencilWrite(self, arg: bool, /) -> None: ...

    @property
    def stencilWriteDepthFail(self) -> bool: ...

    @stencilWriteDepthFail.setter
    def stencilWriteDepthFail(self, arg: bool, /) -> None: ...

    @property
    def stencilCompare(self) -> Material.StencilCompare: ...

    @stencilCompare.setter
    def stencilCompare(self, arg: Material.StencilCompare, /) -> None: ...

    @property
    def stencilMask(self) -> int: ...

    @stencilMask.setter
    def stencilMask(self, arg: int, /) -> None: ...

    @property
    def stencilEffectMode(self) -> Material.StencilEffectMode: ...

    @stencilEffectMode.setter
    def stencilEffectMode(self, arg: Material.StencilEffectMode, /) -> None: ...

    @property
    def stencilEffectColor(self) -> numpy.ndarray: ...

    @stencilEffectColor.setter
    def stencilEffectColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def stencilOutlineThickness(self) -> float: ...

    @stencilOutlineThickness.setter
    def stencilOutlineThickness(self, arg: float, /) -> None: ...

    @property
    def albedoMap(self) -> int: ...

    @albedoMap.setter
    def albedoMap(self, arg: int, /) -> None: ...

    @property
    def foliageTextureAtlasApplied(self) -> bool: ...

    @foliageTextureAtlasApplied.setter
    def foliageTextureAtlasApplied(self, arg: bool, /) -> None: ...

    @property
    def foliageTextureAtlasUvRect(self) -> numpy.ndarray: ...

    @foliageTextureAtlasUvRect.setter
    def foliageTextureAtlasUvRect(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def textureBlendMap(self) -> int: ...

    @textureBlendMap.setter
    def textureBlendMap(self, arg: int, /) -> None: ...

    @property
    def terrainSplatAlbedoMap(self) -> int: ...

    @terrainSplatAlbedoMap.setter
    def terrainSplatAlbedoMap(self, arg: int, /) -> None: ...

    @property
    def terrainSplatControlMap(self) -> int: ...

    @terrainSplatControlMap.setter
    def terrainSplatControlMap(self, arg: int, /) -> None: ...

    @property
    def terrainSplatPrimaryUvScale(self) -> float: ...

    @terrainSplatPrimaryUvScale.setter
    def terrainSplatPrimaryUvScale(self, arg: float, /) -> None: ...

    @property
    def terrainSplatSecondaryUvScale(self) -> float: ...

    @terrainSplatSecondaryUvScale.setter
    def terrainSplatSecondaryUvScale(self, arg: float, /) -> None: ...

    @property
    def terrainSplatControlBounds(self) -> numpy.ndarray: ...

    @terrainSplatControlBounds.setter
    def terrainSplatControlBounds(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def hasTerrainSplat(self) -> bool: ...

    @hasTerrainSplat.setter
    def hasTerrainSplat(self, arg: bool, /) -> None: ...

    @property
    def normalMap(self) -> int: ...

    @normalMap.setter
    def normalMap(self, arg: int, /) -> None: ...

    @property
    def bentNormalMap(self) -> int: ...

    @bentNormalMap.setter
    def bentNormalMap(self, arg: int, /) -> None: ...

    @property
    def metallicMap(self) -> int: ...

    @metallicMap.setter
    def metallicMap(self, arg: int, /) -> None: ...

    @property
    def roughnessMap(self) -> int: ...

    @roughnessMap.setter
    def roughnessMap(self, arg: int, /) -> None: ...

    @property
    def metallicRoughnessMap(self) -> int: ...

    @metallicRoughnessMap.setter
    def metallicRoughnessMap(self, arg: int, /) -> None: ...

    @property
    def ormMap(self) -> int: ...

    @ormMap.setter
    def ormMap(self, arg: int, /) -> None: ...

    @property
    def aoMap(self) -> int: ...

    @aoMap.setter
    def aoMap(self, arg: int, /) -> None: ...

    @property
    def emissiveMap(self) -> int: ...

    @emissiveMap.setter
    def emissiveMap(self, arg: int, /) -> None: ...

    @property
    def clearcoatMap(self) -> int: ...

    @clearcoatMap.setter
    def clearcoatMap(self, arg: int, /) -> None: ...

    @property
    def clearcoatRoughnessMap(self) -> int: ...

    @clearcoatRoughnessMap.setter
    def clearcoatRoughnessMap(self, arg: int, /) -> None: ...

    @property
    def clearcoatNormalMap(self) -> int: ...

    @clearcoatNormalMap.setter
    def clearcoatNormalMap(self, arg: int, /) -> None: ...

    @property
    def sheenColorMap(self) -> int: ...

    @sheenColorMap.setter
    def sheenColorMap(self, arg: int, /) -> None: ...

    @property
    def sheenRoughnessMap(self) -> int: ...

    @sheenRoughnessMap.setter
    def sheenRoughnessMap(self, arg: int, /) -> None: ...

    @property
    def transmissionMap(self) -> int: ...

    @transmissionMap.setter
    def transmissionMap(self, arg: int, /) -> None: ...

    @property
    def refractionMap(self) -> int: ...

    @refractionMap.setter
    def refractionMap(self, arg: int, /) -> None: ...

    @property
    def thicknessMap(self) -> int: ...

    @thicknessMap.setter
    def thicknessMap(self, arg: int, /) -> None: ...

    @property
    def subsurfaceMap(self) -> int: ...

    @subsurfaceMap.setter
    def subsurfaceMap(self, arg: int, /) -> None: ...

    @property
    def subsurfaceTransmittanceMap(self) -> int: ...

    @subsurfaceTransmittanceMap.setter
    def subsurfaceTransmittanceMap(self, arg: int, /) -> None: ...

    @property
    def backlightMap(self) -> int: ...

    @backlightMap.setter
    def backlightMap(self, arg: int, /) -> None: ...

    @property
    def anisotropyMap(self) -> int: ...

    @anisotropyMap.setter
    def anisotropyMap(self, arg: int, /) -> None: ...

    @property
    def anisotropyMapAlphaStrength(self) -> bool: ...

    @anisotropyMapAlphaStrength.setter
    def anisotropyMapAlphaStrength(self, arg: bool, /) -> None: ...

    @property
    def weatherMaskMap(self) -> int: ...

    @weatherMaskMap.setter
    def weatherMaskMap(self, arg: int, /) -> None: ...

    @property
    def rimMap(self) -> int: ...

    @rimMap.setter
    def rimMap(self, arg: int, /) -> None: ...

    @property
    def heightMap(self) -> int: ...

    @heightMap.setter
    def heightMap(self, arg: int, /) -> None: ...

    @property
    def detailMaskMap(self) -> int: ...

    @detailMaskMap.setter
    def detailMaskMap(self, arg: int, /) -> None: ...

    @property
    def detailAlbedoMap(self) -> int: ...

    @detailAlbedoMap.setter
    def detailAlbedoMap(self, arg: int, /) -> None: ...

    @property
    def detailNormalMap(self) -> int: ...

    @detailNormalMap.setter
    def detailNormalMap(self, arg: int, /) -> None: ...

    @property
    def lightmapMap(self) -> int: ...

    @lightmapMap.setter
    def lightmapMap(self, arg: int, /) -> None: ...

    @property
    def environmentMap(self) -> int: ...

    @environmentMap.setter
    def environmentMap(self, arg: int, /) -> None: ...

    @property
    def irradianceMap(self) -> int: ...

    @irradianceMap.setter
    def irradianceMap(self, arg: int, /) -> None: ...

    @property
    def prefilteredEnvironmentMap(self) -> int: ...

    @prefilteredEnvironmentMap.setter
    def prefilteredEnvironmentMap(self, arg: int, /) -> None: ...

    @property
    def brdfLut(self) -> int: ...

    @brdfLut.setter
    def brdfLut(self, arg: int, /) -> None: ...

    @property
    def importReport(self) -> str: ...

    @importReport.setter
    def importReport(self, arg: str, /) -> None: ...

    @staticmethod
    def pbr(name: str, baseColor: numpy.ndarray, metallic: float = 0.0, roughness: float = 0.5) -> Material:
        """
        /**
             * @brief Build a metallic/roughness PBR material with sensible defaults.
             *
             * Sets `type = Type::PBR`, the albedo tint, metallic, and roughness factors;
             * leaves all texture slots empty. Provide an IBL environment separately
             * via `Visuals::setPbrEnvironment(...)` if you want indirect lighting.
             * @param name Material name.
             * @param baseColor Linear RGBA albedo tint.
             * @param metallic Metallic factor in [0, 1].
             * @param roughness Roughness factor in [0, 1].
             * @return The configured material.
             */
        C++: Material (std::string, const glm::vec4 &, float, float)
        """

    @staticmethod
    def glass(name: str = 'Glass', thickness: float = 0.0, tint: numpy.ndarray = ..., attenuationDistance: float = 1.0, roughness: float = 0.03999999910593033, indexOfRefraction: float = 1.5) -> Material:
        """
        /**
             * @brief Dielectric glass. Zero thickness represents a thin window; positive
             * thickness is an authored world-space optical path length for a solid.
             * Tint is the transmitted color after attenuationDistance world units.
             * Keep alpha at one: transmission, not coverage, makes glass transparent.
             * Enable RenderQualitySettings::screenSpaceRefraction for scene transmission.
             * Sets transmissionFactor = 1 and metallic = 0; a thin pane (thickness <= 0)
             * is also made double-sided.
             * @param name Material name.
             * @param thickness Optical path length in world units (clamped to >= 0).
             * @param tint Transmitted color, clamped to [0, 1].
             * @param attenuationDistance Distance at which @p tint is reached (min 1e-4).
             * @param roughness Surface roughness.
             * @param indexOfRefraction IOR, clamped to [1, 3].
             * @return The configured material.
             */
        C++: Material (std::string, float, const glm::vec3 &, float, float, float)
        """

    def isTransmissive(self) -> bool:
        """
        /// Optical transmission of this pass (does not traverse nextPass).
            /// @return True if transmission or refraction factor is positive or its map is set.
        C++: bool () const
        """

    def hasTransmissivePass(self) -> bool:
        """
        /// @return True when this pass or any nextPass isTransmissive().
        C++: bool () const
        """

    def hasProximityFadePass(self) -> bool:
        """
        /// @return True when this pass or any nextPass enables proximity fade.
        C++: bool () const
        """

    def usesDefaultDepthAndStencilState(self) -> bool:
        """
        /// @return True when this pass and every nextPass render with the pass's own depth test/write and
            /// no stencil or decal depth handling.
        C++: bool () const
        """

    @staticmethod
    def unlitColor(name: str, color: numpy.ndarray = ...) -> Material:
        """
        /**
             * @brief Build an unlit material that ignores scene lighting.
             *
             * Useful for HUD elements, debug visuals, holograms, light-emitting decals.
             * Named `unlitColor` rather than `unlit` to avoid colliding with the
             * `bool unlit` data member. Sets #unlit, which selects the PBR shader's
             * unlit path.
             * @param name Material name.
             * @param color Linear RGBA color.
             * @return The configured material.
             */
        C++: Material (std::string, const glm::vec4 &)
        """

    @staticmethod
    def simpleColor(name: str, color: numpy.ndarray) -> Material:
        """
        /**
             * @brief Build a simple-color (non-PBR) material that responds to lighting.
             *
             * Routes through the simple mesh shader (sets #forceSimpleShading). Cheaper
             * than `pbr()` and appropriate for primitives that do not need
             * metallic/roughness or IBL.
             * @param name Material name.
             * @param color Linear RGBA color.
             * @return The configured material.
             */
        C++: Material (std::string, const glm::vec4 &)
        """

    @staticmethod
    def foliage(name: str, foliageKind: Material.FoliageType, baseColor: numpy.ndarray) -> Material:
        """
        /**
             * @brief Build a foliage material with the given type and base color.
             *
             * Sets #foliageType, a PBR type with roughness 0.85 and metallic 0, and
             * #doubleSided (no culling, two-sided lighting). Other foliage fields keep
             * their defaults; in particular #foliageWindStrength stays 0, so there is no
             * wind deformation until it is set.
             * @param name Material name.
             * @param foliageKind Foliage category.
             * @param baseColor Linear RGBA albedo tint.
             * @return The configured material.
             */
        C++: Material (std::string, FoliageType, const glm::vec4 &)
        """

    @staticmethod
    def defaultGround(name: str = 'GroundMaterial') -> Material:
        """
        /**
             * @brief Build a neutral PBR ground material (mid-grey, rough, non-metallic).
             *
             * Convenient default for the world's ground plane / heightmap when no
             * authored material is supplied. Base color 0.55 grey, roughness 0.9, metallic 0.
             * @param name Material name.
             * @return The configured material.
             */
        C++: Material (std::string)
        """

    def applyTo(self, shader: Shader) -> None:
        """
        /**
             * @brief Applies the material's properties to a shader.
             *
             * This method binds the necessary textures and sets all material-related
             * uniforms on the provided shader before a draw call. It also makes
             * @p shader the active program. Requires a current GL context.
             *
             * @param shader The shader to which the properties will be applied.
             */
        C++: void (Shader &) const
        """

    def setShadingModel(self, model: Material.ShadingModel) -> None:
        """
        /**
             * @brief Write `diffuseMode`, `specularMode`, and `unlit` from a high-level model.
             *
             * Helper for the common case where callers want "PBR/Lambert/Toon/Unlit"
             * rather than picking a `DiffuseMode` + `SpecularMode` pair and remembering
             * to clear the `unlit` flag. `Unlit` only sets #unlit and leaves the BRDF modes unchanged.
             * @param model Desired shading model.
             */
        C++: void (ShadingModel)
        """

    def shadingModel(self) -> Material.ShadingModel:
        """
        /**
             * @brief Read back a best-effort high-level shading model classification.
             *
             * Returns `Unlit` when the `unlit` flag is set, otherwise picks the closest
             * `Standard`/`Lambert`/`Toon` based on the current `diffuseMode` /
             * `specularMode`. A Toon diffuse or specular mode yields `Toon`; Burley
             * diffuse yields `Standard`; anything else yields `Lambert`.
             * @return The classification.
             */
        C++: ShadingModel () const
        """

    def alphaSettings(self) -> Material.AlphaSettings:
        """
        /**
             * @brief Bundle the three alpha-related enums into a single struct view.
             *
             * The returned struct is a snapshot; round-trip it via `setAlphaSettings()`
             * to write back into the underlying fields.
             * @return Copy of the alpha fields.
             */
        C++: AlphaSettings () const
        """

    def setAlphaSettings(self, s: Material.AlphaSettings) -> None:
        """
        /// @brief Write `alphaMode`, `blendMode`, `alphaAntialiasing` from one struct.
            /// @param s Values to write.
        C++: void (const AlphaSettings &)
        """

    def foliageSettings(self) -> Material.FoliageSettings:
        """
        /// @return Copy of the foliage fields.
        C++: FoliageSettings () const
        """

    def setFoliageSettings(self, f: Material.FoliageSettings) -> None:
        """
        /// Write all foliage fields at once.
            /// @param f Values to write.
        C++: void (const FoliageSettings &)
        """

    def parallaxSettings(self) -> Material.ParallaxSettings:
        """
        /// @return Copy of the parallax fields.
        C++: ParallaxSettings () const
        """

    def setParallaxSettings(self, p: Material.ParallaxSettings) -> None:
        """
        /// Write all parallax fields at once.
            /// @param p Values to write.
        C++: void (const ParallaxSettings &)
        """

    def vertexDisplacementSettings(self) -> Material.VertexDisplacementSettings:
        """
        /// @return Copy of the vertex-displacement fields.
        C++: VertexDisplacementSettings () const
        """

    def setVertexDisplacementSettings(self, p: Material.VertexDisplacementSettings) -> None:
        """
        /// Write both vertex-displacement fields.
            /// @param p Values to write.
        C++: void (const VertexDisplacementSettings &)
        """

    def usesPbrShading(self) -> bool:
        """
        /**
             * @brief Decide between the simple and the PBR shader.
             * @return False when #forceSimpleShading is set; otherwise true for Type::PBR,
             *         a next pass, or any field that only the PBR shader implements.
             */
        C++: bool () const
        """

    def canUseCorePbr(self) -> bool:
        """
        /**
             * @return True when this material (and every next pass) can be drawn by the compact core PBR shader without losing authored features.
             */
        C++: bool () const
        """

    def canUseFoliageFastPbr(self) -> bool:
        """
        /**
             * @return True when this material (and every next pass) can use the compact alpha-masked foliage PBR shader.
             */
        C++: bool () const
        """

    def receivesActiveWeather(self) -> bool:
        """
        /**
             * @brief True when this material actually picks up weather (wetness/snow).
             *
             * Mirrors the per-material weather gate in the full PBR fragment shader: a
             * non-PBR or non-porous material, or one that opts out of both wetness and
             * snow, receives no weather effect — so it does not require the
             * weather-capable full shader even when weather is globally enabled. Used by
             * the scene-level shader selection so a per-material weather opt-out actually
             * preserves the core/foliage-fast path (it previously did not).
             * @return True when any pass in the chain receives weather.
             */
        C++: bool () const
        """

    def canUseCompactPbr(self) -> bool:
        """
        /**
             * @return True when either compact PBR shader can reproduce this material.
             */
        C++: bool () const
        """

    def requiresHighFidelityPbr(self) -> bool:
        """
        /**
             * @return True when this PBR material (or a next pass) uses transmission, volume, subsurface,
             *         backlight, anisotropy, iridescence or sheen, which benefit from the high-fidelity PBR shader.
             */
        C++: bool () const
        """

    def needsTransparentPass(self) -> bool:
        """
        /**
             * @return True when this material or a next pass must be sorted and rendered by the transparent pass.
             */
        C++: bool () const
        """

    def hasDepthTestOverride(self) -> bool:
        """
        /**
             * @return True when this material explicitly overrides the current depth-test state.
             */
        C++: bool () const
        """

    def hasDepthWriteOverride(self) -> bool:
        """
        /**
             * @return True when this material explicitly overrides the current depth-write state.
             */
        C++: bool () const
        """

    def usesInvertedDepthTest(self) -> bool:
        """
        /**
             * @return True when this material wants an inverted depth comparison.
             */
        C++: bool () const
        """

    def hasStencilOverride(self) -> bool:
        """
        /**
             * @return True when #stencilReference is non-negative.
             */
        C++: bool () const
        """

    def resolveDepthTest(self, passDefault: bool) -> bool:
        """
        /**
             * @brief Resolve material depth-test override against a render-pass default.
             * @param passDefault Depth-test state of the current pass.
             * @return Whether depth testing is enabled (Inverted counts as enabled).
             */
        C++: bool (bool) const
        """

    def resolveDepthWrite(self, passDefault: bool) -> bool:
        """
        /**
             * @brief Resolve material depth-write override against a render-pass default.
             * @param passDefault Depth-write state of the current pass.
             * @return Whether depth writes are enabled.
             */
        C++: bool (bool) const
        """

    def usesTwoSidedLighting(self) -> bool:
        """
        /**
             * @return True when back-facing fragments should use two-sided lighting
             *         (#doubleSided or CullMode::Disabled).
             */
        C++: bool () const
        """

    @staticmethod
    def textureBindingTable() -> list[Material.TextureBindingInfo]:
        """
        /**
             * @brief Stable material sampler binding table used by Material::applyTo().
             * @return Static table with one entry per bindable map (including IBL cubemaps).
             */
        C++: const std::array<TextureBindingInfo, 32> &()
        """

    @staticmethod
    def textureSlotName(slot: Material.TextureSlot) -> int | None:
        """
        /// @param slot Texture slot.
            /// @return Slot name (e.g. "albedo"), or "unknown" for out-of-range values.
        C++: const char *(TextureSlot)
        """

    @overload
    def textureUvTransform(self, slot: Material.TextureSlot) -> Material.UvTransform:
        """
        /// @param slot Texture slot; must not be TextureSlot::Count (not bounds-checked).
            /// @return Mutable reference into #textureUvTransforms.
        C++: UvTransform &(TextureSlot)
        """

    @overload
    def textureUvTransform(self, slot: Material.TextureSlot) -> Material.UvTransform:
        """
        /// @param slot Texture slot; must not be TextureSlot::Count (not bounds-checked).
            /// @return Reference into #textureUvTransforms.
        C++: const UvTransform &(TextureSlot) const
        """

    def hasPerTextureUvTransforms(self) -> bool:
        """
        /// @return True when any slot has an authored UV transform.
        C++: bool () const
        """

    def perTextureUvTransformCount(self) -> int:
        """
        /// @return Number of slots with an authored UV transform.
        C++: size_t () const
        """

    @staticmethod
    def textureSlotInfo(slot: Material.TextureSlot) -> Material.TextureSlotInfo:
        """
        /**
             * @brief Look up metadata for a texture slot by enum.
             *
             * @param slot Texture slot.
             * @return Metadata describing the slot. `Count` is invalid and returns
             *         an entry named "unknown" with both flags false.
             */
        C++: const TextureSlotInfo &(TextureSlot)
        """

class MeshPreprocessCacheStats:
    def __init__(self) -> None: ...

    @property
    def metadataLoads(self) -> int: ...

    @metadataLoads.setter
    def metadataLoads(self, arg: int, /) -> None: ...

    @property
    def metadataStores(self) -> int: ...

    @metadataStores.setter
    def metadataStores(self, arg: int, /) -> None: ...

    @property
    def metadataMisses(self) -> int: ...

    @metadataMisses.setter
    def metadataMisses(self, arg: int, /) -> None: ...

    @property
    def lodLoads(self) -> int: ...

    @lodLoads.setter
    def lodLoads(self, arg: int, /) -> None: ...

    @property
    def lodStores(self) -> int: ...

    @lodStores.setter
    def lodStores(self, arg: int, /) -> None: ...

    @property
    def lodMisses(self) -> int: ...

    @lodMisses.setter
    def lodMisses(self, arg: int, /) -> None: ...

    @property
    def fallbackWarnings(self) -> int: ...

    @fallbackWarnings.setter
    def fallbackWarnings(self, arg: int, /) -> None: ...

class MeshPreprocessLodFlavor(enum.Enum):
    Visual = 0

    Instanced = 1

    FoliagePartitioned = 2

    Foliage = 3

class MeshPreprocessMetadata:
    def __init__(self) -> None: ...

    @property
    def valid(self) -> bool: ...

    @valid.setter
    def valid(self, arg: bool, /) -> None: ...

    @property
    def vertexCount(self) -> int: ...

    @vertexCount.setter
    def vertexCount(self, arg: int, /) -> None: ...

    @property
    def triangleCount(self) -> int: ...

    @triangleCount.setter
    def triangleCount(self, arg: int, /) -> None: ...

    @property
    def localAabbValid(self) -> bool: ...

    @localAabbValid.setter
    def localAabbValid(self, arg: bool, /) -> None: ...

    @property
    def localAabbMin(self) -> numpy.ndarray: ...

    @localAabbMin.setter
    def localAabbMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def localAabbMax(self) -> numpy.ndarray: ...

    @localAabbMax.setter
    def localAabbMax(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def localBoundsValid(self) -> bool: ...

    @localBoundsValid.setter
    def localBoundsValid(self, arg: bool, /) -> None: ...

    @property
    def localBoundsCenter(self) -> numpy.ndarray: ...

    @localBoundsCenter.setter
    def localBoundsCenter(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def localBoundsRadius(self) -> float: ...

    @localBoundsRadius.setter
    def localBoundsRadius(self, arg: float, /) -> None: ...

    @property
    def hasAlphaMaskedOrFoliageMaterial(self) -> bool: ...

    @hasAlphaMaskedOrFoliageMaterial.setter
    def hasAlphaMaskedOrFoliageMaterial(self, arg: bool, /) -> None: ...

    @property
    def hasTransparentMaterial(self) -> bool: ...

    @hasTransparentMaterial.setter
    def hasTransparentMaterial(self, arg: bool, /) -> None: ...

    @property
    def hasSpecialBlendMode(self) -> bool: ...

    @hasSpecialBlendMode.setter
    def hasSpecialBlendMode(self, arg: bool, /) -> None: ...

    @property
    def hasPlanarReflectionMeshes(self) -> bool: ...

    @hasPlanarReflectionMeshes.setter
    def hasPlanarReflectionMeshes(self, arg: bool, /) -> None: ...

    @property
    def maxRenderPriority(self) -> int: ...

    @maxRenderPriority.setter
    def maxRenderPriority(self, arg: int, /) -> None: ...

class PbrEnvironment:
    def __init__(self) -> None: ...

    class LoadOptions:
        def __init__(self) -> None: ...

        @property
        def environmentFaceSize(self) -> int: ...

        @environmentFaceSize.setter
        def environmentFaceSize(self, arg: int, /) -> None: ...

        @property
        def irradianceFaceSize(self) -> int: ...

        @irradianceFaceSize.setter
        def irradianceFaceSize(self, arg: int, /) -> None: ...

        @property
        def irradianceSampleCount(self) -> int: ...

        @irradianceSampleCount.setter
        def irradianceSampleCount(self, arg: int, /) -> None: ...

        @property
        def prefilteredFaceSize(self) -> int: ...

        @prefilteredFaceSize.setter
        def prefilteredFaceSize(self, arg: int, /) -> None: ...

        @property
        def prefilteredMipLevels(self) -> int: ...

        @prefilteredMipLevels.setter
        def prefilteredMipLevels(self, arg: int, /) -> None: ...

        @property
        def prefilteredSampleCount(self) -> int: ...

        @prefilteredSampleCount.setter
        def prefilteredSampleCount(self, arg: int, /) -> None: ...

        @property
        def brdfLutSize(self) -> int: ...

        @brdfLutSize.setter
        def brdfLutSize(self, arg: int, /) -> None: ...

        @property
        def brdfLutSampleCount(self) -> int: ...

        @brdfLutSampleCount.setter
        def brdfLutSampleCount(self, arg: int, /) -> None: ...

        @property
        def strength(self) -> float: ...

        @strength.setter
        def strength(self, arg: float, /) -> None: ...

        @property
        def generateEnvironmentMipmaps(self) -> bool: ...

        @generateEnvironmentMipmaps.setter
        def generateEnvironmentMipmaps(self, arg: bool, /) -> None: ...

    @property
    def environmentCubemap(self) -> int: ...

    @environmentCubemap.setter
    def environmentCubemap(self, arg: int, /) -> None: ...

    @property
    def irradianceCubemap(self) -> int: ...

    @irradianceCubemap.setter
    def irradianceCubemap(self, arg: int, /) -> None: ...

    @property
    def prefilteredCubemap(self) -> int: ...

    @prefilteredCubemap.setter
    def prefilteredCubemap(self, arg: int, /) -> None: ...

    @property
    def splitSumBrdfLut(self) -> int: ...

    @splitSumBrdfLut.setter
    def splitSumBrdfLut(self, arg: int, /) -> None: ...

    @property
    def prefilteredMipCount(self) -> int: ...

    @prefilteredMipCount.setter
    def prefilteredMipCount(self, arg: int, /) -> None: ...

    @property
    def strength(self) -> float: ...

    @strength.setter
    def strength(self, arg: float, /) -> None: ...

    @property
    def boxProjection(self) -> bool: ...

    @boxProjection.setter
    def boxProjection(self, arg: bool, /) -> None: ...

    @property
    def probePosition(self) -> numpy.ndarray: ...

    @probePosition.setter
    def probePosition(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def boxMin(self) -> numpy.ndarray: ...

    @boxMin.setter
    def boxMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def boxMax(self) -> numpy.ndarray: ...

    @boxMax.setter
    def boxMax(self, arg: numpy.ndarray, /) -> None: ...

    def isComplete(self) -> bool:
        """
        /** @return True when all four texture handles are non-zero. */
        C++: bool () const
        """

    @overload
    @staticmethod
    def loadFromHdrFile(hdrPath: str) -> PbrEnvironment:
        """
        /**
           * @brief Build a complete PbrEnvironment from a single Radiance .hdr file.
           *
           * Loads the equirectangular cubemap, integrates the diffuse irradiance,
           * computes the GGX-prefiltered specular cubemap, and creates the split-sum
           * BRDF LUT. All four handles are owned by the active GL context and remain
           * valid for the lifetime of that context.
           *
           * Each handle that fails to build (for example because the file cannot be
           * read) is 0, and `isComplete()` then returns false. Handles that did build,
           * such as the file-independent BRDF LUT, are still returned.
           *
           * Defined inline in RayraiWindow.hpp so the static cubemap creators are
           * visible.
           *
           * @param hdrPath Path to the equirectangular Radiance .hdr file.
           * @return The baked environment; check isComplete() before use.
           */
        C++: PbrEnvironment (const std::string &)
        """

    @overload
    @staticmethod
    def loadFromHdrFile(hdrPath: str, opts: PbrEnvironment.LoadOptions) -> PbrEnvironment:
        """
        /// @copydoc loadFromHdrFile(const std::string&)
          /// @param opts Face sizes, sample counts and strength used for the bake.
        C++: PbrEnvironment (const std::string &, const LoadOptions &)
        """

class PhysicsLinkPose:
    def __init__(self) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def rotation(self) -> numpy.ndarray: ...

    @rotation.setter
    def rotation(self, arg: numpy.ndarray, /) -> None: ...

class PhysicsMeshSnapshot:
    def __init__(self) -> None: ...

    @property
    def revision(self) -> int: ...

    @revision.setter
    def revision(self, arg: int, /) -> None: ...

    @property
    def topologyRevision(self) -> int: ...

    @topologyRevision.setter
    def topologyRevision(self, arg: int, /) -> None: ...

    @property
    def blocks(self) -> list[list[Vertex]]: ...

    @blocks.setter
    def blocks(self, arg: Sequence[Sequence[Vertex]], /) -> None: ...

    @property
    def indices(self) -> list[int]: ...

    @indices.setter
    def indices(self, arg: Sequence[int], /) -> None: ...

    @property
    def samples(self) -> numpy.ndarray: ...

    @samples.setter
    def samples(self, arg: numpy.ndarray, /) -> None: ...

    def vertexCount(self) -> int:
        """C++: size_t () const"""

    def vertex(self, index: int) -> Vertex:
        """C++: const Vertex &(size_t) const"""

    def vertices(self) -> list[Vertex]:
        """
        /// All vertices, in one array.
        C++: std::vector<Vertex> () const
        """

class PhysicsObjectSnapshot:
    def __init__(self) -> None: ...

    @property
    def id(self) -> int: ...

    @id.setter
    def id(self, arg: int, /) -> None: ...

    @property
    def worldIndex(self) -> int: ...

    @worldIndex.setter
    def worldIndex(self, arg: int, /) -> None: ...

    @property
    def type(self) -> raisimpy.ObjectType: ...

    @type.setter
    def type(self, arg: raisimpy.ObjectType, /) -> None: ...

    @property
    def bodyType(self) -> raisimpy.BodyType: ...

    @bodyType.setter
    def bodyType(self, arg: raisimpy.BodyType, /) -> None: ...

    @property
    def revision(self) -> int: ...

    @revision.setter
    def revision(self, arg: int, /) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def rotation(self) -> numpy.ndarray: ...

    @rotation.setter
    def rotation(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def description(self) -> PhysicsRenderDescription: ...

    @description.setter
    def description(self, arg: PhysicsRenderDescription, /) -> None: ...

    @property
    def geometry(self) -> PhysicsMeshSnapshot: ...

    @geometry.setter
    def geometry(self, arg: PhysicsMeshSnapshot, /) -> None: ...

    @property
    def links(self) -> list[PhysicsLinkPose]: ...

    @links.setter
    def links(self, arg: Sequence[PhysicsLinkPose], /) -> None: ...

class PhysicsRenderDescription:
    def __init__(self) -> None: ...

    @property
    def geometryRevision(self) -> int: ...

    @geometryRevision.setter
    def geometryRevision(self, arg: int, /) -> None: ...

    @property
    def appearanceRevision(self) -> int: ...

    @appearanceRevision.setter
    def appearanceRevision(self, arg: int, /) -> None: ...

    @property
    def appearance(self) -> str: ...

    @appearance.setter
    def appearance(self, arg: str, /) -> None: ...

    @property
    def visualCount(self) -> int: ...

    @visualCount.setter
    def visualCount(self, arg: int, /) -> None: ...

    @property
    def geometry(self) -> list[PhysicsShapeSnapshot]: ...

    @geometry.setter
    def geometry(self, arg: Sequence[PhysicsShapeSnapshot], /) -> None: ...

    def shapes(self) -> list[PhysicsShapeSnapshot]:
        """C++: const std::vector<PhysicsShapeSnapshot> &() const"""

    @property
    def colors(self) -> list[numpy.ndarray]: ...

    @colors.setter
    def colors(self, arg: Sequence[numpy.ndarray], /) -> None: ...

    @property
    def segmentationLabels(self) -> list[int]: ...

    @segmentationLabels.setter
    def segmentationLabels(self, arg: Sequence[int], /) -> None: ...

class PhysicsShapeSnapshot:
    def __init__(self) -> None: ...

    @property
    def shape(self) -> raisimpy.ShapeType: ...

    @shape.setter
    def shape(self, arg: raisimpy.ShapeType, /) -> None: ...

    @property
    def dimensions(self) -> numpy.ndarray: ...

    @dimensions.setter
    def dimensions(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def scale(self) -> numpy.ndarray: ...

    @scale.setter
    def scale(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def offset(self) -> numpy.ndarray: ...

    @offset.setter
    def offset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def rotation(self) -> numpy.ndarray: ...

    @rotation.setter
    def rotation(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def localModel(self) -> numpy.ndarray: ...

    @localModel.setter
    def localModel(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def localIdx(self) -> int: ...

    @localIdx.setter
    def localIdx(self, arg: int, /) -> None: ...

    @property
    def fileName(self) -> str: ...

    @fileName.setter
    def fileName(self, arg: str, /) -> None: ...

    @property
    def resourceDirectory(self) -> str: ...

    @resourceDirectory.setter
    def resourceDirectory(self, arg: str, /) -> None: ...

class PhysicsSnapshot:
    def __init__(self) -> None: ...

    @property
    def simTime(self) -> float: ...

    @simTime.setter
    def simTime(self, arg: float, /) -> None: ...

    @property
    def configuration(self) -> int: ...

    @configuration.setter
    def configuration(self, arg: int, /) -> None: ...

    @property
    def topologyRevision(self) -> int: ...

    @topologyRevision.setter
    def topologyRevision(self, arg: int, /) -> None: ...

    @property
    def objects(self) -> list[PhysicsObjectSnapshot]: ...

    @objects.setter
    def objects(self, arg: Sequence[PhysicsObjectSnapshot], /) -> None: ...

    @property
    def tendons(self) -> list[PhysicsTendonSnapshot]: ...

    @tendons.setter
    def tendons(self, arg: Sequence[PhysicsTendonSnapshot], /) -> None: ...

    @property
    def explosions(self) -> list[raisimpy.Explosion]: ...

    @explosions.setter
    def explosions(self, arg: Sequence[raisimpy.Explosion], /) -> None: ...

    @property
    def projectiles(self) -> list[raisimpy.Projectile]: ...

    @projectiles.setter
    def projectiles(self, arg: Sequence[raisimpy.Projectile], /) -> None: ...

    @property
    def projectileHits(self) -> list[raisimpy.ProjectileHit]: ...

    @projectileHits.setter
    def projectileHits(self, arg: Sequence[raisimpy.ProjectileHit], /) -> None: ...

    @property
    def launches(self) -> list[raisimpy.Projectile]: ...

    @launches.setter
    def launches(self, arg: Sequence[raisimpy.Projectile], /) -> None: ...

    @property
    def wind(self) -> numpy.ndarray: ...

    @wind.setter
    def wind(self, arg: numpy.ndarray, /) -> None: ...

class PhysicsSnapshotCapture:
    def __init__(self) -> None: ...

    def capture(self, world: raisimpy.World) -> PhysicsSnapshot:
        """C++: std::shared_ptr<const PhysicsSnapshot> (raisim::World &)"""

class PhysicsTendonSnapshot:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def color(self) -> numpy.ndarray: ...

    @color.setter
    def color(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def instances(self) -> list["raisim::server::TendonVisualInstance"]: ...

    @instances.setter
    def instances(self, arg: Sequence["raisim::server::TendonVisualInstance"], /) -> None: ...

class PointCloud:
    def __init__(self, viewer: RayraiWindow) -> None: ...

    @property
    def positions(self) -> list[numpy.ndarray]: ...

    @positions.setter
    def positions(self, arg: Sequence[numpy.ndarray], /) -> None: ...

    @property
    def colors(self) -> list[numpy.ndarray]: ...

    @colors.setter
    def colors(self, arg: Sequence[numpy.ndarray], /) -> None: ...

    @property
    def pointSize(self) -> float: ...

    @pointSize.setter
    def pointSize(self, arg: float, /) -> None: ...

    def setCategory(self, category: VisualCategory) -> None:
        """
        /**
           * @brief Set the render category used by capture/detection filters.
           *
           * Point clouds are visible in the normal interactive viewer regardless of this
           * category. External-camera and generated-capture passes can request only
           * detectable point clouds, in which case clouds left as
           * `VisualCategory::NotDetectable` are skipped. This lets debug lidar traces,
           * helper samples, and temporary overlays remain visible to a human operator
           * without appearing in external-camera or generated capture images.
           *
           * This is render metadata only; it does not create collision, physics
           * behavior, or semantic labels. New `PointCloud` instances default to
           * `VisualCategory::NotDetectable`.
           *
           * @param category New render category.
           * @see VisualCategory
           */
        C++: void (VisualCategory)
        """

    def setDetectable(self, detectable: bool) -> None:
        """
        /**
           * @brief Convenience helper for toggling the point cloud render category.
           * @param detectable True to include this cloud in detectable-only
           *        capture/detection render passes; false to keep it viewer-only for
           *        those passes.
           */
        C++: void (bool)
        """

    def getCategory(self) -> VisualCategory:
        """
        /**
           * @brief Get the render category used by capture/detection filters.
           * @return Current point cloud category.
           */
        C++: VisualCategory () const
        """

    def isDetectable(self) -> bool:
        """
        /**
           * @brief Check whether detectable-only render passes include this point cloud.
           * @return True when the category is `VisualCategory::Detectable`.
           */
        C++: bool () const
        """

    def enabled(self) -> bool:
        """
        /// @return True when rendering is enabled.
        C++: bool () const
        """

    def renderRevision(self) -> int:
        """
        /// Internal: @return Counter bumped on every render-affecting change, used by
          ///        the renderer to invalidate cached frames.
        C++: uint64_t () const
        """

    def updatePointBuffer(self) -> None:
        """
        /**
           * @brief Upload positions and colors to the GPU (or free the buffers if empty).
           *
           * Also recomputes the bounding sphere. Does nothing (and logs an error) when
           * `colors.size() != positions.size()`.
           */
        C++: void ()
        """

    def updatePointPositionsBuffer(self) -> None:
        """
        /**
           * @brief Streaming variant for point clouds whose colors stay unchanged.
           *        Updates bounds + position VBO only and reuses the initialized color buffer.
           */
        C++: void ()
        """

    def updatePointRange(self, begin: int, count: int) -> None:
        """
        /**
           * @brief Upload a changed point range of positions and colors in place.
           *
           * No-op unless the buffers were already uploaded with matching sizes; the
           * range is clamped to the uploaded capacity. Bounds are not recomputed, so
           * call updatePointBuffer() if points moved outside the previous bounds.
           * @param begin Index of the first changed point.
           * @param count Number of changed points.
           */
        C++: void (size_t, size_t)
        """

    def draw(self, camera: Camera, shader: Shader) -> None:
        """
        /**
           * @brief Draw the point cloud.
           * @param camera Active camera (unused but kept for consistency).
           * @param shader Shader to use for drawing; must already be in use (only the
           *        `pointSize` uniform is set here).
           */
        C++: void (Camera &, Shader &)
        """

    def approximateBounds(self) -> tuple:
        """
        /**
           * @brief Bounding sphere (centroid + max distance) of the last uploaded positions.
           * @param[out] center Sphere center in world coordinates.
           * @param[out] radius Sphere radius.
           * @return False (outputs untouched) when #positions is empty.
           */
        C++: bool (glm::vec3 &, float &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def setMaxRenderedPoints(self, maxPoints: int) -> None:
        """
        /**
           * @brief Cap drawn points for simple single-threaded LOD.
           * @param maxPoints Maximum number of points drawn per frame; 0 draws all.
           */
        C++: void (size_t)
        """

    def maxRenderedPoints(self) -> int:
        """
        /// @return Point cap set by setMaxRenderedPoints() (0 = unlimited).
        C++: size_t () const
        """

    def setRenderedPointStride(self, stride: int) -> None:
        """
        /**
           * @brief Draw every Nth point for cheap point-cloud LOD.
           * @param stride Point stride; 0 is treated as 1 (draw contiguous points).
           */
        C++: void (size_t)
        """

    def renderedPointStride(self) -> int:
        """
        /// @return Current point stride (>= 1).
        C++: size_t () const
        """

    def enable(self, enable: bool) -> None:
        """
        /**
           * @brief Enable or disable rendering.
           * @param enable True to enable.
           */
        C++: void (bool)
        """

    def notifyDataChanged(self) -> None:
        """
        /**
           * @brief Bump the render revision after a direct mutation of `positions` or `colors`.
           *
           * `setCategory()`, `enable()`, `setMaxRenderedPoints()` already bump the
           * revision themselves; this method is for the public `positions` /
           * `colors` vectors, which the setter pattern does not cover.
           */
        C++: void ()
        """

class PointShadowPolicy(enum.Enum):
    Disabled = 0

    StandaloneCubemap = 1

class PostProcessDebugMode(enum.Enum):
    Final = 0

    BloomSource = 1

    AmbientOcclusion = 2

    AmbientOcclusionRaw = 3

    FogTransmittance = 4

    FogDensity = 5

    FogScattering = 6

    WorldNormal = 7

    DepthHeatmap = 8

    ScreenSpaceIndirectLighting = 9

class ProjectedDecal:
    def __init__(self) -> None: ...

    @property
    def center(self) -> numpy.ndarray: ...

    @center.setter
    def center(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def halfExtents(self) -> numpy.ndarray: ...

    @halfExtents.setter
    def halfExtents(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def orientation(self) -> numpy.ndarray: ...

    @orientation.setter
    def orientation(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def color(self) -> numpy.ndarray: ...

    @color.setter
    def color(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def edgeFade(self) -> float: ...

    @edgeFade.setter
    def edgeFade(self, arg: float, /) -> None: ...

    @property
    def normalFade(self) -> float: ...

    @normalFade.setter
    def normalFade(self, arg: float, /) -> None: ...

    @property
    def upperFade(self) -> float: ...

    @upperFade.setter
    def upperFade(self, arg: float, /) -> None: ...

    @property
    def lowerFade(self) -> float: ...

    @lowerFade.setter
    def lowerFade(self, arg: float, /) -> None: ...

    @property
    def albedoMap(self) -> int: ...

    @albedoMap.setter
    def albedoMap(self, arg: int, /) -> None: ...

    @property
    def emissionMap(self) -> int: ...

    @emissionMap.setter
    def emissionMap(self, arg: int, /) -> None: ...

    @property
    def normalMap(self) -> int: ...

    @normalMap.setter
    def normalMap(self, arg: int, /) -> None: ...

    @property
    def ormMap(self) -> int: ...

    @ormMap.setter
    def ormMap(self, arg: int, /) -> None: ...

    @property
    def textureUvScaleOffset(self) -> numpy.ndarray: ...

    @textureUvScaleOffset.setter
    def textureUvScaleOffset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def albedoMix(self) -> float: ...

    @albedoMix.setter
    def albedoMix(self, arg: float, /) -> None: ...

    @property
    def emissionEnergy(self) -> float: ...

    @emissionEnergy.setter
    def emissionEnergy(self, arg: float, /) -> None: ...

    @property
    def normalStrength(self) -> float: ...

    @normalStrength.setter
    def normalStrength(self, arg: float, /) -> None: ...

    @property
    def ormStrength(self) -> float: ...

    @ormStrength.setter
    def ormStrength(self, arg: float, /) -> None: ...

    @property
    def distanceFadeEnabled(self) -> bool: ...

    @distanceFadeEnabled.setter
    def distanceFadeEnabled(self, arg: bool, /) -> None: ...

    @property
    def distanceFadeBegin(self) -> float: ...

    @distanceFadeBegin.setter
    def distanceFadeBegin(self, arg: float, /) -> None: ...

    @property
    def distanceFadeLength(self) -> float: ...

    @distanceFadeLength.setter
    def distanceFadeLength(self, arg: float, /) -> None: ...

class RaisimObject:
    def __init__(self, viewer: RayraiWindow, snapshot: PhysicsObjectSnapshot, parent: RayraiWindow | None, asset: RayraiGlobalAsset) -> None: ...

    class Changes:
        def __init__(self) -> None: ...

        @property
        def revision(self) -> bool: ...

        @revision.setter
        def revision(self, arg: bool, /) -> None: ...

        @property
        def candidate(self) -> bool: ...

        @candidate.setter
        def candidate(self, arg: bool, /) -> None: ...

        @property
        def shadow(self) -> bool: ...

        @shadow.setter
        def shadow(self, arg: bool, /) -> None: ...

    def apply(self, snapshot: PhysicsObjectSnapshot) -> RaisimObject.Changes:
        """
        /**
           * @brief Advance to a newer snapshot of the same object.
           *
           * Uploads changed mesh vertices (a vertex delta when the snapshot provides
           * one) and rebuilds meshes whose shapes changed. An unchanged revision costs a
           * comparison. Requires a current GL context.
           * @param snapshot Object state with the same object identity, copied when it changed.
           * @return What changed.
           */
        C++: Changes (const PhysicsObjectSnapshot &)
        """

    def sourceObject(self) -> raisimpy.Object | None:
        """
        /// @name Snapshot state
          ///@{
          /// @return The source object, as an identity key only; it may no longer exist.
        C++: raisim::Object *() const
        """

    def instanceId(self) -> int:
        """
        /// @return An id no other RaisimObject in the process had, so a drawable built for another
          ///         object at the same address is told apart.
        C++: uint64_t () const
        """

    def objectType(self) -> raisimpy.ObjectType:
        """C++: raisim::ObjectType () const"""

    def bodyType(self) -> raisimpy.BodyType:
        """C++: raisim::BodyType () const"""

    def objectId(self) -> int:
        """C++: raisim::Object::Id () const"""

    def worldIndex(self) -> int:
        """
        /// @return Index of the object in World::getObjList() when its snapshot was captured.
        C++: int32_t () const
        """

    def segmentationLabel(self) -> int:
        """
        /// @return Segmentation label of the object (of its first shape for articulated systems).
        C++: int32_t () const
        """

    def renderRevision(self) -> int:
        """
        /// @return Revision that advances whenever the drawn state changes.
        C++: uint64_t () const
        """

    def shadowGeometrySignature(self, showCollisionBodies: bool) -> int:
        """
        /// Internal: depth-affecting mesh readiness and material state of the drawn geometry.
        C++: uint64_t (bool) const
        """

    def worldPosition(self) -> numpy.ndarray:
        """
        /// @return World position of the object origin (the plane height for a ground).
        C++: glm::vec3 () const
        """

    def worldRotation(self) -> numpy.ndarray:
        """
        /// @return World rotation of the object frame.
        C++: glm::mat3 () const
        """

    def draw(self, shader: Shader, showCollisionBodies: bool, mode: OpenGLMesh.RenderMode = OpenGLMesh.RenderMode.All) -> None:
        """
        /**
           * @brief Draw the object using the provided shader.
           *
           * Sets the `shadingType`, `model` and `objectColor` uniforms per mesh, and
           * `uSegmentationId` (world index, body index, label, 0) when the shader has
           * it; the shader must already be in use. No-op when hidden.
           * @param shader Shader used for rendering.
           * @param showCollisionBodies True to render collision geometry when available
           *        (articulated systems only).
           * @param mode Which material passes of each mesh to draw.
           */
        C++: void (Shader &, bool, OpenGLMesh::RenderMode)
        """

    def hasPbrMeshes(self) -> bool:
        """
        /**
           * @name Material queries
           * Internal: renderer pass-selection queries over all cached meshes.
           */
          ///@{
          /**
           * @return True when any cached mesh needs the PBR shader.
           */
        C++: bool () const
        """

    def hasHighFidelityPbrMeshes(self) -> bool:
        """
        /// @return True when any PBR mesh requires the high-fidelity PBR path (Material::requiresHighFidelityPbr()).
        C++: bool () const
        """

    def hasTransmissivePbrMeshes(self) -> bool:
        """
        /// @return True when any PBR mesh needs the transmission pass (Material::hasTransmissivePass()).
        C++: bool () const
        """

    def hasProximityFadePbrMeshes(self) -> bool:
        """
        /// @return True when any PBR mesh uses proximity fade (Material::hasProximityFadePass()).
        C++: bool () const
        """

    def usesDefaultDepthAndStencilState(self) -> bool:
        """
        /// @return True when every mesh uses default depth/stencil state (Material::usesDefaultDepthAndStencilState()).
        C++: bool () const
        """

    def hasFullPbrMeshes(self) -> bool:
        """
        /// @return True when any PBR mesh cannot use the compact PBR shader (Material::canUseCompactPbr()).
        C++: bool () const
        """

    def hasFoliageFastPbrMeshes(self) -> bool:
        """
        /// @return True when any mesh can use the foliage fast-PBR path (Material::canUseFoliageFastPbr()).
        C++: bool () const
        """

    def hasWeatherEligiblePbrMeshes(self) -> bool:
        """
        /// @return True when any mesh reacts to active weather (Material::receivesActiveWeather()).
        C++: bool () const
        """

    def hasSimpleMeshes(self) -> bool:
        """
        /// @return True when any mesh uses non-PBR (simple/textured) shading.
        C++: bool () const
        """

    def hasPlanarReflectionMeshes(self) -> bool:
        """
        /// @return True when any PBR mesh enables Material::usePlanarReflection.
        C++: bool () const
        """

    def hasTransparentMaterials(self) -> bool:
        """
        /**
           * @return True when any cached mesh material must be rendered in the transparent pass.
           */
        C++: bool () const
        """

    def hasSpecialBlendMode(self) -> bool:
        """
        /**
           * @return True when any cached mesh uses a blend mode other than Material::BlendMode::Mix.
           */
        C++: bool () const
        """

    def renderPriority(self) -> int:
        """
        /**
           * @return Highest Material::renderPriority across cached meshes (0 if none).
           */
        C++: int () const
        """

    def isSelectable(self) -> bool:
        """
        ///@}
          /**
           * @brief Check if this object can be selected in the picker.
           * @return True if not hidden, and not a heightmap or ground plane.
           */
        C++: bool () const
        """

    def configureGroundMaterial(self, reflective: bool, roughness: float, metallic: float, proceduralChecker: bool = False) -> None:
        """
        /**
           * @brief Configure ground and heightmap terrain materials. No-op for other objects.
           *
           * For a ground plane, resets its material and makes it PBR with planar
           * reflection when @p reflective, or simple/textured otherwise. For a
           * heightmap, only the window's terrain overrides are reapplied and the
           * other arguments are ignored.
           * @param reflective Use a reflective PBR ground.
           * @param roughness PBR roughness, clamped to [0.04, 1] (reflective only).
           * @param metallic PBR metallic factor, clamped to [0, 1] (reflective only).
           * @param proceduralChecker Draw a procedural checker instead of the ground texture.
           */
        C++: void (bool, float, float, bool)
        """

    def setHidden(self, h: bool) -> None:
        """
        /**
           * @brief Hide or show this object in rendering, independent of its appearance.
           *
           * The object is drawn only while neither this nor a "hidden" appearance hides it.
           * @param h True to hide.
           */
        C++: void (bool)
        """

    def hidden(self) -> bool:
        """
        /**
           * @brief Check if object is hidden.
           * @return True if hidden by setHidden() or by its appearance.
           */
        C++: bool () const
        """

    def getColor(self) -> numpy.ndarray:
        """
        /**
           * @brief Get object color used for rendering.
           * @return RGBA color in [0, 1] parsed from the raisim appearance string
           *         (0.7 grey by default). A negative alpha on a mesh object means
           *         "use the mesh's own colors".
           */
        C++: glm::vec4 () const
        """

    def approximateRadius(self, showCollisionBodies: bool) -> float:
        """
        /**
           * @brief Conservative bounding radius for coarse culling.
           * @param showCollisionBodies Measure collision instead of visual geometry.
           * @return Radius from approximateBounds(), or 1 when bounds are unavailable.
           */
        C++: float (bool) const
        """

    def shadowGeometryWithinBounds(self) -> bool:
        """
        /// @return True when shadow depth draws keep every vertex inside approximateBounds():
          ///         no material uses billboards, fixed-size scaling or height displacement.
        C++: bool () const
        """

    def shadowDepthMayDiscard(self, showCollisionBodies: bool, transparentShadowDithering: bool) -> bool:
        """
        /// @return True when a shadow depth draw of this object can discard fragments: a mesh that
          ///         draw() draws has an alpha-mask material, or an alpha-blend material while
          ///         @p transparentShadowDithering is set. Deformable surfaces always report true.
        C++: bool (bool, bool) const
        """

    def approximateBounds(self, showCollisionBodies: bool) -> tuple:
        """
        /**
           * @brief Conservative world-space bounds for coarse culling.
           *
           * The center is allowed to differ from the raisim object origin for offset
           * meshes, articulated links, and deformables.
           * @param showCollisionBodies Measure collision instead of visual geometry.
           * @param[out] center Sphere center in world coordinates.
           * @param[out] radius Sphere radius in metres.
           * @return False when the object has no geometry.
           */
        C++: bool (bool, glm::vec3 &, float &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def cullingBounds(self, showCollisionBodies: bool) -> tuple:
        """
        /**
           * @brief Bounds for coarse culling: approximateBounds(), or a unit sphere at the
           *        object origin when the object draws no geometry.
           * @param showCollisionBodies Measure collision instead of visual geometry.
           * @param[out] center Sphere center in world coordinates.
           * @param[out] radius Sphere radius in metres.
           */
        C++: void (bool, glm::vec3 &, float &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def getPrimitiveBatchData(self, showCollisionBodies: bool) -> tuple:
        """
        /**
           * @brief Return transform/color data for simple single-body primitive batching.
           * @param showCollisionBodies Batching is refused when true.
           * @param[out] model World model matrix including the primitive scale.
           * @param[out] shape Primitive shape (sphere, box, cylinder or capsule).
           * @param[out] color Object color.
           * @return True when this object can be drawn by the internal primitive batch path
           *         (visible, opaque, single sphere/box/cylinder/capsule).
           */
        C++: bool (bool, glm::mat4 &, raisim::Shape::Type &, glm::vec4 &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def getPrimitiveBatchTemplate(self, showCollisionBodies: bool) -> tuple:
        """
        /**
           * @brief Pose-independent part of getPrimitiveBatchData().
           * @param showCollisionBodies Batching is refused when true.
           * @param[out] shape Primitive shape.
           * @param[out] scale Primitive scale.
           * @param[out] color Object color.
           * @return True when this object is batchable.
           */
        C++: bool (bool, raisim::Shape::Type &, glm::vec3 &, glm::vec4 &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def setDefaultPrimitivePbrEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Toggle High/Ultra default PBR shading for built-in primitive meshes.
           *
           * Box/sphere/cylinder meshes come from the shared asset cache, so this
           * changes their material for every object that shares them.
           * @param enabled True to use the default PBR primitive material.
           */
        C++: void (bool)
        """

    def collectMaterialTextureIds(self) -> tuple:
        """
        /**
           * @brief Append non-environment material texture ids used by this object's cached meshes.
           * @param[in,out] textureIds Ids are appended (duplicates are not removed here).
           */
        C++: void (std::vector<unsigned int> &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def accumulateGeometryDiagnostics(self, diagnostics: OpenGLMesh.GeometryDiagnostics, tinyTriangleAreaThreshold: float) -> None:
        """
        /**
           * @brief Accumulate CPU-side mesh geometry diagnostics for this object.
           * @param[in,out] diagnostics Totals to add to.
           * @param tinyTriangleAreaThreshold Triangles below this area count as tiny.
           */
        C++: void (OpenGLMesh::GeometryDiagnostics &, float) const
        """

    def collectMeshDiagnostics(self, tinyTriangleAreaThreshold: float) -> tuple:
        """
        /**
           * @brief Append per-mesh geometry/material diagnostics for this object.
           * @param[in,out] diagnostics One entry is appended per mesh.
           * @param tinyTriangleAreaThreshold Triangles below this area count as tiny.
           */
        C++: void (std::vector<OpenGLMesh::MeshDiagnostics> &, float) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def debugModelMatrixCacheRebuilds(self) -> int:
        """
        /**
           * @return Number of model-matrix rebuilds, for tests and diagnostics.
           */
        C++: size_t () const
        """

    def debugMeshLists(self) -> list[list[OpenGLMesh]]:
        """
        /// Internal: @return The cached mesh lists (tests/diagnostics).
        C++: const std::vector<std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>>> &() const
        """

    def debugHasMeshGeometry(self) -> bool:
        """
        /// Internal: @return True when any cached mesh list is non-empty (tests/diagnostics).
        C++: bool () const
        """

class RayraiGlobalAsset:
    def __init__(self, viewer: RayraiWindow) -> None: ...

    class MeshLodChain:
        def __init__(self, viewer: RayraiWindow) -> None: ...

        @property
        def levels(self) -> list[list[OpenGLMesh]]: ...

        @levels.setter
        def levels(self, arg: Sequence[Sequence[OpenGLMesh]], /) -> None: ...

        @property
        def triangleCounts(self) -> list[int]: ...

        @triangleCounts.setter
        def triangleCounts(self, arg: Sequence[int], /) -> None: ...

    class SharedGpuAssetStats:
        def __init__(self) -> None: ...

        @property
        def cacheHits(self) -> int: ...

        @cacheHits.setter
        def cacheHits(self, arg: int, /) -> None: ...

        @property
        def cacheMisses(self) -> int: ...

        @cacheMisses.setter
        def cacheMisses(self, arg: int, /) -> None: ...

        @property
        def gpuBufferBytesUploaded(self) -> int: ...

        @gpuBufferBytesUploaded.setter
        def gpuBufferBytesUploaded(self, arg: int, /) -> None: ...

        @property
        def gpuBufferBytesReused(self) -> int: ...

        @gpuBufferBytesReused.setter
        def gpuBufferBytesReused(self, arg: int, /) -> None: ...

        @property
        def cachedSnapshots(self) -> int: ...

        @cachedSnapshots.setter
        def cachedSnapshots(self, arg: int, /) -> None: ...

    def getMeshes(self, name: str) -> list[OpenGLMesh]:
        """
        /**
           * @brief Retrieves a model's meshes by its asset name (usually file path).
           *
           * Assets are keyed by canonical path, so different spellings of the same
           * file share one mesh list. With async loading enabled (the default; USD
           * files always load synchronously) a new file returns an empty list at once
           * that is filled in place by a later pollAsyncLoads(); if that import fails
           * the list stays empty. The cache holds only weak references.
           * Requires a current GL context for synchronous loads.
           * @param name The unique identifier for the asset, typically the file path.
           * @return Shared mesh list (possibly still empty while loading), or nullptr
           *         if the path does not exist or a synchronous load fails.
           */
        C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> (const std::string &)
        """

    def getMeshesWithCanonicalKey(self, name: str) -> tuple:
        """
        /**
           * @brief Retrieves meshes and returns the canonical cache key used for the lookup.
           * @param name The unique identifier for the asset, typically the file path.
           * @param canonicalKeyOut Optional output for the canonical mesh key ("" if
           *        the path cannot be resolved).
           * @return Same as getMeshes(const std::string&).
           */
        C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> (const std::string &, std::string *)
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def findLoadedMeshes(self, name: str) -> list[OpenGLMesh]:
        """
        /**
           * @brief The meshes of a model file this cache has already read, without reading it.
           * @param name File path, as passed to getMeshes().
           * @return The cached mesh list (possibly still empty while an asynchronous read
           *         runs), or nullptr when the file was not read or nothing holds it any more.
           */
        C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> (const std::string &) const
        """

    def canonicalMeshKey(self, path: str) -> str:
        """
        /**
           * @brief Resolve a mesh path to the canonical key used by getMeshes() / the LOD cache.
           * @param path File path; results are memoized per input string.
           * @return Canonical absolute path, or "" if the path cannot be canonicalized
           *         (does not exist, etc.).
           */
        C++: std::string (const std::string &) const
        """

    def getCachedMeshLods(self, canonicalKey: str) -> RayraiGlobalAsset.MeshLodChain:
        """
        /**
           * @brief Look up a cached automatic-LOD chain.
           * @param canonicalKey Key from canonicalMeshKey().
           * @return The chain, or nullptr on miss or when it has expired.
           */
        C++: std::shared_ptr<MeshLodChain> (const std::string &) const
        """

    def cacheMeshLods(self, canonicalKey: str, chain: RayraiGlobalAsset.MeshLodChain) -> None:
        """
        /**
           * @brief Register an automatic-LOD chain for a canonical source-mesh key.
           *        The cache holds weak references, so the chain stays alive only while
           *        at least one caller (typically a Visuals) keeps the returned shared_ptr.
           * @param canonicalKey Key from canonicalMeshKey(); empty keys are ignored.
           * @param chain Chain to register; null is ignored.
           */
        C++: void (const std::string &, const std::shared_ptr<MeshLodChain> &)
        """

    def getCachedMeshPreprocessMetadata(self, canonicalKey: str) -> MeshPreprocessMetadata:
        """
        /**
           * @brief Look up file-backed mesh preprocessing metadata.
           * @param canonicalKey Key from canonicalMeshKey().
           * @return The metadata, or nullptr on miss or when it has expired.
           */
        C++: std::shared_ptr<MeshPreprocessMetadata> (const std::string &) const
        """

    def cacheMeshPreprocessMetadata(self, canonicalKey: str, metadata: MeshPreprocessMetadata) -> None:
        """
        /**
           * @brief Register file-backed mesh preprocessing metadata for later visuals.
           *
           * Held by weak reference, like cacheMeshLods().
           * @param canonicalKey Key from canonicalMeshKey(); empty keys are ignored.
           * @param metadata Metadata to register; null or invalid metadata is ignored.
           */
        C++: void (const std::string &, const std::shared_ptr<MeshPreprocessMetadata> &)
        """

    def pollAsyncLoads(self, maxAssets: int = 1) -> int:
        """
        /**
           * @brief Advance ready import, worker geometry, and incremental GPU upload stages.
           *
           * This must be called on the OpenGL/render thread because it creates textures,
           * vertex buffers, and vertex array objects. Each processed asset advances one
           * stage (import, geometry preparation, or one submesh upload) per call.
           * Throws std::logic_error as requireUploadContext() does, before advancing any.
           * @param maxAssets Maximum number of pending assets advanced in this call.
           * @return Number of assets finalized (successfully or not) in this call.
           */
        C++: size_t (size_t)
        """

    def pollAsyncLoadsUntil(self, deadline: datetime.timedelta | float) -> int:
        """
        /**
           * @brief Advance every pending import as far as it can go before @p deadline.
           *
           * Like pollAsyncLoads(), on the OpenGL/render thread, but an asset keeps
           * advancing (one submesh upload after another) until it waits for a worker or
           * the deadline passes. Imports whose worker finished go first, so the next
           * worker stages start early. With async loading enabled, an import is
           * finished only once the background decodes of its textures are done, so the
           * render thread decodes no image. At least one ready stage runs per call.
           * Throws std::logic_error as requireUploadContext() does, before advancing any.
           * @param deadline Time after which no further stage starts.
           * @return Number of assets finalized (successfully or not) in this call.
           */
        C++: size_t (std::chrono::steady_clock::time_point)
        """

    def pendingAsyncLoadCount(self) -> int:
        """
        /**
           * @return Number of mesh file imports that have not been finalized yet.
           */
        C++: size_t () const
        """

    def requireUploadContext(self) -> None:
        """
        /**
           * @brief Fails unless the OpenGL context current at construction is current.
           *
           * The store's textures, vertex arrays and staging regions belong to that
           * context, so its loads advance only there. Elsewhere every pending load
           * first gives its staged copies back to the staging buffer (it stages again,
           * or uploads directly, once polled in its context), and then this throws
           * std::logic_error.
           */
        C++: void ()
        """

    def setAsyncMeshLoadingEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Enable or disable async mesh file import for future getMeshes() calls.
           *
           * Loads already pending keep loading asynchronously (they wait for the background
           * decodes of their textures and copy through the staging buffer), unless a
           * synchronous getMeshes() waits for one of them or loadPendingSynchronously() is called.
           * @param enabled True (default) to import on a worker thread; finalize with pollAsyncLoads().
           */
        C++: void (bool)
        """

    def loadPendingSynchronously(self) -> None:
        """
        /**
           * @brief Makes every pending load finish synchronously as it is polled: it no longer waits
           *        for the background decodes of its textures or for room in the staging buffer.
           *
           * RayraiWindow::setAsyncMeshLoadingEnabled(false) calls this before finishing them.
           */
        C++: void ()
        """

    def forgetChangedModelFiles(self) -> None:
        """
        /**
           * @brief Forget every cached model file whose sources changed since it was read.
           *
           * A model's sources are its file and every file its import read (glTF buffers, an
           * OBJ material library, ...). A changed one makes the next getMeshes() read the model
           * again, even while visuals still hold the meshes read before. RayraiWindow::swapWorld()
           * calls this, so a rebuilt scene shows edited files.
           */
        C++: void ()
        """

    def releaseRecentlyReadModels(self) -> None:
        """
        /**
           * @brief Release the models read from files since the last call.
           *
           * A model read from a file stays referenced until this is called, so every visual
           * created in the same frame shares one read, even when a visual keeps only meshes
           * derived from it (instanced foliage keeps its partitioned meshes, and a terrain
           * foliage layer adds one visual per chunk). RayraiWindow calls this at the end of
           * each update(), renderWithExternalCamera(), and depth-only render.
           */
        C++: void ()
        """

    def asyncMeshLoadingEnabled(self) -> bool:
        """
        /**
           * @return True when future mesh file imports start on a worker thread.
           */
        C++: bool () const
        """

    def getCubeMesh(self) -> list[OpenGLMesh]:
        """
        /**
           * @brief Get a unit cube mesh (side length 1, centred at the origin).
           * @return Shared mesh list for a cube.
           */
        C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> ()
        """

    def getCylinderMesh(self) -> list[OpenGLMesh]:
        """
        /**
           * @brief Get a unit cylinder mesh (radius 1, height 1 along local Z).
           * @return Shared mesh list for a cylinder.
           */
        C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> ()
        """

    def getCapsuleMesh(self) -> list[OpenGLMesh]:
        """
        /**
           * @brief Get a unit capsule mesh (radius 1, cylinder height 1, along local Z).
           * @return Shared mesh list for a capsule.
           */
        C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> ()
        """

    def getSphereMesh(self) -> list[OpenGLMesh]:
        """
        /**
           * @brief Get a unit sphere mesh (radius 1).
           * @return Shared mesh list for a sphere.
           */
        C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> ()
        """

    def getPlaneMesh(self) -> list[OpenGLMesh]:
        """
        /**
           * @brief Get a unit plane mesh (1 x 1, centred on the XY plane).
           * @return Shared mesh list for a plane.
           */
        C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> ()
        """

    @staticmethod
    def configureDefaultPrimitivePbrMaterial(meshes: Sequence[OpenGLMesh], enabled: bool) -> None:
        """
        /**
           * @brief Configure the default material used by built-in primitive meshes.
           *
           * The material remains a PBR material, but @p enabled controls whether it is
           * allowed to enter the PBR pass or forced through the cheaper simple shader.
           * Overwrites the base color, metallic, roughness and IBL strengths of every
           * mesh in @p meshes.
           * @param meshes Mesh list to modify in place; null is ignored.
           * @param enabled False sets Material::forceSimpleShading.
           */
        C++: void (const std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> &, bool)
        """

    def setDefaultPrimitivePbrEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Enable High/Ultra primitive PBR defaults for shared primitive assets.
           *
           * Applies configureDefaultPrimitivePbrMaterial() to the cube, sphere,
           * cylinder and capsule meshes (not the plane).
           * @param enabled True to allow the PBR pass.
           */
        C++: void (bool)
        """

    def defaultPrimitivePbrEnabled(self) -> bool:
        """
        /**
           * @return True when shared primitive assets currently use the High/Ultra PBR path.
           */
        C++: bool () const
        """

    @staticmethod
    def sharedGpuAssetStats() -> RayraiGlobalAsset.SharedGpuAssetStats:
        """
        /**
           * @brief Process-wide shared GPU-asset cache diagnostics.
           *
           * Byte counts cover indexed mesh VBO/EBO allocations. Textures and
           * context-local VAOs are intentionally excluded. Thread-safe.
           * @return Snapshot of the counters.
           */
        C++: SharedGpuAssetStats ()
        """

    @staticmethod
    def resetSharedGpuAssetStats() -> None:
        """
        /// Reset the process-wide counters to zero (cached assets are kept). Thread-safe.
        C++: void ()
        """

    @staticmethod
    def setSharedGpuAssetCacheEnabled(enabled: bool) -> None:
        """
        /**
           * @brief Enable process-wide reuse of immutable mesh buffers across compatible contexts.
           *
           * Compatibility is probed with OpenGL object visibility, so contexts outside
           * the same share group automatically retain independent uploads. Enabled by default.
           * @param enabled True to reuse uploads; false makes every lookup upload anew.
           */
        C++: void (bool)
        """

    @staticmethod
    def sharedGpuAssetCacheEnabled() -> bool:
        """
        /// @return True when process-wide mesh-buffer reuse is enabled.
        C++: bool ()
        """

class RenderQualityPreset(enum.Enum):
    Fast = 0

    Balanced = 1

    High = 2

    Ultra = 3

    Custom = 4

class RenderQualitySettings:
    def __init__(self) -> None: ...

    class BloomSettings:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def threshold(self) -> float: ...

        @threshold.setter
        def threshold(self, arg: float, /) -> None: ...

        @property
        def strength(self) -> float: ...

        @strength.setter
        def strength(self, arg: float, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def knee(self) -> float: ...

        @knee.setter
        def knee(self, arg: float, /) -> None: ...

        @property
        def quality(self) -> int: ...

        @quality.setter
        def quality(self, arg: int, /) -> None: ...

        @property
        def sourceClamp(self) -> float: ...

        @sourceClamp.setter
        def sourceClamp(self, arg: float, /) -> None: ...

        @property
        def anamorphic(self) -> float: ...

        @anamorphic.setter
        def anamorphic(self, arg: float, /) -> None: ...

        @property
        def dirtStrength(self) -> float: ...

        @dirtStrength.setter
        def dirtStrength(self, arg: float, /) -> None: ...

        @property
        def dirtScale(self) -> float: ...

        @dirtScale.setter
        def dirtScale(self, arg: float, /) -> None: ...

        @property
        def dirtTexture(self) -> int: ...

        @dirtTexture.setter
        def dirtTexture(self, arg: int, /) -> None: ...

        @property
        def blendMode(self) -> GlowBlendMode: ...

        @blendMode.setter
        def blendMode(self, arg: GlowBlendMode, /) -> None: ...

        @property
        def mix(self) -> float: ...

        @mix.setter
        def mix(self, arg: float, /) -> None: ...

    class DepthOfFieldSettings:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def focusDistance(self) -> float: ...

        @focusDistance.setter
        def focusDistance(self, arg: float, /) -> None: ...

        @property
        def focusRange(self) -> float: ...

        @focusRange.setter
        def focusRange(self, arg: float, /) -> None: ...

        @property
        def maxRadius(self) -> float: ...

        @maxRadius.setter
        def maxRadius(self, arg: float, /) -> None: ...

    class FogSettings:
        def __init__(self) -> None: ...

        @property
        def density(self) -> float: ...

        @density.setter
        def density(self, arg: float, /) -> None: ...

        @property
        def colorOverrideEnabled(self) -> bool: ...

        @colorOverrideEnabled.setter
        def colorOverrideEnabled(self, arg: bool, /) -> None: ...

        @property
        def color(self) -> numpy.ndarray: ...

        @color.setter
        def color(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def heightEnabled(self) -> bool: ...

        @heightEnabled.setter
        def heightEnabled(self, arg: bool, /) -> None: ...

        @property
        def heightDensity(self) -> float: ...

        @heightDensity.setter
        def heightDensity(self, arg: float, /) -> None: ...

        @property
        def heightBaseHeight(self) -> float: ...

        @heightBaseHeight.setter
        def heightBaseHeight(self, arg: float, /) -> None: ...

        @property
        def heightFalloff(self) -> float: ...

        @heightFalloff.setter
        def heightFalloff(self, arg: float, /) -> None: ...

        @property
        def volumetricEnabled(self) -> bool: ...

        @volumetricEnabled.setter
        def volumetricEnabled(self, arg: bool, /) -> None: ...

        @property
        def volumetricDensity(self) -> float: ...

        @volumetricDensity.setter
        def volumetricDensity(self, arg: float, /) -> None: ...

        @property
        def volumetricNoiseScale(self) -> float: ...

        @volumetricNoiseScale.setter
        def volumetricNoiseScale(self, arg: float, /) -> None: ...

        @property
        def volumetricNoiseStrength(self) -> float: ...

        @volumetricNoiseStrength.setter
        def volumetricNoiseStrength(self, arg: float, /) -> None: ...

        @property
        def volumetricColor(self) -> numpy.ndarray: ...

        @volumetricColor.setter
        def volumetricColor(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def volumetricAnisotropy(self) -> float: ...

        @volumetricAnisotropy.setter
        def volumetricAnisotropy(self, arg: float, /) -> None: ...

    class ShadowsSettings:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def resolution(self) -> int: ...

        @resolution.setter
        def resolution(self, arg: int, /) -> None: ...

        @property
        def bias(self) -> float: ...

        @bias.setter
        def bias(self, arg: float, /) -> None: ...

        @property
        def strength(self) -> float: ...

        @strength.setter
        def strength(self, arg: float, /) -> None: ...

        @property
        def pcfRadius(self) -> float: ...

        @pcfRadius.setter
        def pcfRadius(self, arg: float, /) -> None: ...

        @property
        def contactHardening(self) -> bool: ...

        @contactHardening.setter
        def contactHardening(self, arg: bool, /) -> None: ...

        @property
        def contactHardeningMinRadius(self) -> float: ...

        @contactHardeningMinRadius.setter
        def contactHardeningMinRadius(self, arg: float, /) -> None: ...

        @property
        def contactHardeningMaxRadius(self) -> float: ...

        @contactHardeningMaxRadius.setter
        def contactHardeningMaxRadius(self, arg: float, /) -> None: ...

        @property
        def contactHardeningDepthScale(self) -> float: ...

        @contactHardeningDepthScale.setter
        def contactHardeningDepthScale(self, arg: float, /) -> None: ...

        @property
        def orthoHalfSize(self) -> float: ...

        @orthoHalfSize.setter
        def orthoHalfSize(self, arg: float, /) -> None: ...

        @property
        def nearPlane(self) -> float: ...

        @nearPlane.setter
        def nearPlane(self, arg: float, /) -> None: ...

        @property
        def farPlane(self) -> float: ...

        @farPlane.setter
        def farPlane(self, arg: float, /) -> None: ...

        @property
        def centerOffset(self) -> float: ...

        @centerOffset.setter
        def centerOffset(self, arg: float, /) -> None: ...

        @property
        def directionalCascadeCount(self) -> int: ...

        @directionalCascadeCount.setter
        def directionalCascadeCount(self, arg: int, /) -> None: ...

        @property
        def directionalCascadeLambda(self) -> float: ...

        @directionalCascadeLambda.setter
        def directionalCascadeLambda(self, arg: float, /) -> None: ...

        @property
        def directionalCascadeMaxDistance(self) -> float: ...

        @directionalCascadeMaxDistance.setter
        def directionalCascadeMaxDistance(self, arg: float, /) -> None: ...

        @property
        def updateEveryFrame(self) -> bool: ...

        @updateEveryFrame.setter
        def updateEveryFrame(self, arg: bool, /) -> None: ...

        @property
        def directionalSoftnessDegrees(self) -> float: ...

        @directionalSoftnessDegrees.setter
        def directionalSoftnessDegrees(self, arg: float, /) -> None: ...

        @property
        def directionalSoftnessMaxRadius(self) -> float: ...

        @directionalSoftnessMaxRadius.setter
        def directionalSoftnessMaxRadius(self, arg: float, /) -> None: ...

    class TemporalAaSettings:
        def __init__(self) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def blend(self) -> float: ...

        @blend.setter
        def blend(self, arg: float, /) -> None: ...

        @property
        def jitterScale(self) -> float: ...

        @jitterScale.setter
        def jitterScale(self, arg: float, /) -> None: ...

        @property
        def method(self) -> TemporalAaMethod: ...

        @method.setter
        def method(self, arg: TemporalAaMethod, /) -> None: ...

        @property
        def currentWeight(self) -> float: ...

        @currentWeight.setter
        def currentWeight(self, arg: float, /) -> None: ...

        @property
        def sharpness(self) -> float: ...

        @sharpness.setter
        def sharpness(self, arg: float, /) -> None: ...

    class ToneMappingSettings:
        def __init__(self) -> None: ...

        @property
        def pbrToneMappingEnabled(self) -> bool: ...

        @pbrToneMappingEnabled.setter
        def pbrToneMappingEnabled(self, arg: bool, /) -> None: ...

        @property
        def exposure(self) -> float: ...

        @exposure.setter
        def exposure(self, arg: float, /) -> None: ...

        @property
        def autoExposureEnabled(self) -> bool: ...

        @autoExposureEnabled.setter
        def autoExposureEnabled(self, arg: bool, /) -> None: ...

        @property
        def autoExposureKey(self) -> float: ...

        @autoExposureKey.setter
        def autoExposureKey(self, arg: float, /) -> None: ...

        @property
        def autoExposureSpeed(self) -> float: ...

        @autoExposureSpeed.setter
        def autoExposureSpeed(self, arg: float, /) -> None: ...

        @property
        def autoExposureMinFactor(self) -> float: ...

        @autoExposureMinFactor.setter
        def autoExposureMinFactor(self, arg: float, /) -> None: ...

        @property
        def autoExposureMaxFactor(self) -> float: ...

        @autoExposureMaxFactor.setter
        def autoExposureMaxFactor(self, arg: float, /) -> None: ...

        @property
        def whiteBalance(self) -> numpy.ndarray: ...

        @whiteBalance.setter
        def whiteBalance(self, arg: numpy.ndarray, /) -> None: ...

        @property
        def saturation(self) -> float: ...

        @saturation.setter
        def saturation(self, arg: float, /) -> None: ...

        @property
        def gradePreset(self) -> ColorGradePreset: ...

        @gradePreset.setter
        def gradePreset(self, arg: ColorGradePreset, /) -> None: ...

        @property
        def gradeStrength(self) -> float: ...

        @gradeStrength.setter
        def gradeStrength(self, arg: float, /) -> None: ...

    @property
    def backgroundColorRgb255(self) -> numpy.ndarray: ...

    @backgroundColorRgb255.setter
    def backgroundColorRgb255(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def mainLightAmbient(self) -> numpy.ndarray: ...

    @mainLightAmbient.setter
    def mainLightAmbient(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def mainLightDiffuse(self) -> numpy.ndarray: ...

    @mainLightDiffuse.setter
    def mainLightDiffuse(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def mainLightSpecular(self) -> numpy.ndarray: ...

    @mainLightSpecular.setter
    def mainLightSpecular(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def mainLightDirection(self) -> numpy.ndarray: ...

    @mainLightDirection.setter
    def mainLightDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def shadowResolution(self) -> int: ...

    @shadowResolution.setter
    def shadowResolution(self, arg: int, /) -> None: ...

    @property
    def shadowBias(self) -> float: ...

    @shadowBias.setter
    def shadowBias(self, arg: float, /) -> None: ...

    @property
    def shadowStrength(self) -> float: ...

    @shadowStrength.setter
    def shadowStrength(self, arg: float, /) -> None: ...

    @property
    def shadowPcfRadius(self) -> float: ...

    @shadowPcfRadius.setter
    def shadowPcfRadius(self, arg: float, /) -> None: ...

    @property
    def directionalShadowSoftnessDegrees(self) -> float: ...

    @directionalShadowSoftnessDegrees.setter
    def directionalShadowSoftnessDegrees(self, arg: float, /) -> None: ...

    @property
    def directionalShadowSoftnessMaxRadius(self) -> float: ...

    @directionalShadowSoftnessMaxRadius.setter
    def directionalShadowSoftnessMaxRadius(self, arg: float, /) -> None: ...

    @property
    def shadowContactHardening(self) -> bool: ...

    @shadowContactHardening.setter
    def shadowContactHardening(self, arg: bool, /) -> None: ...

    @property
    def shadowContactHardeningMinRadius(self) -> float: ...

    @shadowContactHardeningMinRadius.setter
    def shadowContactHardeningMinRadius(self, arg: float, /) -> None: ...

    @property
    def shadowContactHardeningMaxRadius(self) -> float: ...

    @shadowContactHardeningMaxRadius.setter
    def shadowContactHardeningMaxRadius(self, arg: float, /) -> None: ...

    @property
    def shadowContactHardeningDepthScale(self) -> float: ...

    @shadowContactHardeningDepthScale.setter
    def shadowContactHardeningDepthScale(self, arg: float, /) -> None: ...

    @property
    def shadowDebugOverlayEnabled(self) -> bool: ...

    @shadowDebugOverlayEnabled.setter
    def shadowDebugOverlayEnabled(self, arg: bool, /) -> None: ...

    @property
    def shadowOrthoHalfSize(self) -> float: ...

    @shadowOrthoHalfSize.setter
    def shadowOrthoHalfSize(self, arg: float, /) -> None: ...

    @property
    def shadowNear(self) -> float: ...

    @shadowNear.setter
    def shadowNear(self, arg: float, /) -> None: ...

    @property
    def shadowFar(self) -> float: ...

    @shadowFar.setter
    def shadowFar(self, arg: float, /) -> None: ...

    @property
    def shadowCenterOffset(self) -> float: ...

    @shadowCenterOffset.setter
    def shadowCenterOffset(self, arg: float, /) -> None: ...

    @property
    def fogDensity(self) -> float: ...

    @fogDensity.setter
    def fogDensity(self, arg: float, /) -> None: ...

    @property
    def fogColorOverrideEnabled(self) -> bool: ...

    @fogColorOverrideEnabled.setter
    def fogColorOverrideEnabled(self, arg: bool, /) -> None: ...

    @property
    def fogColor(self) -> numpy.ndarray: ...

    @fogColor.setter
    def fogColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def heightFogEnabled(self) -> bool: ...

    @heightFogEnabled.setter
    def heightFogEnabled(self, arg: bool, /) -> None: ...

    @property
    def heightFogDensity(self) -> float: ...

    @heightFogDensity.setter
    def heightFogDensity(self, arg: float, /) -> None: ...

    @property
    def heightFogBaseHeight(self) -> float: ...

    @heightFogBaseHeight.setter
    def heightFogBaseHeight(self, arg: float, /) -> None: ...

    @property
    def heightFogFalloff(self) -> float: ...

    @heightFogFalloff.setter
    def heightFogFalloff(self, arg: float, /) -> None: ...

    @property
    def volumetricFogEnabled(self) -> bool: ...

    @volumetricFogEnabled.setter
    def volumetricFogEnabled(self, arg: bool, /) -> None: ...

    @property
    def volumetricFogDensity(self) -> float: ...

    @volumetricFogDensity.setter
    def volumetricFogDensity(self, arg: float, /) -> None: ...

    @property
    def volumetricFogNoiseScale(self) -> float: ...

    @volumetricFogNoiseScale.setter
    def volumetricFogNoiseScale(self, arg: float, /) -> None: ...

    @property
    def volumetricFogNoiseStrength(self) -> float: ...

    @volumetricFogNoiseStrength.setter
    def volumetricFogNoiseStrength(self, arg: float, /) -> None: ...

    @property
    def volumetricFogColor(self) -> numpy.ndarray: ...

    @volumetricFogColor.setter
    def volumetricFogColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def volumetricFogAnisotropy(self) -> float: ...

    @volumetricFogAnisotropy.setter
    def volumetricFogAnisotropy(self, arg: float, /) -> None: ...

    @property
    def volumetricFogAnimationTimeSeconds(self) -> float: ...

    @volumetricFogAnimationTimeSeconds.setter
    def volumetricFogAnimationTimeSeconds(self, arg: float, /) -> None: ...

    @property
    def volumetricFogWindDirection(self) -> numpy.ndarray: ...

    @volumetricFogWindDirection.setter
    def volumetricFogWindDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def volumetricFogWindSpeed(self) -> float: ...

    @volumetricFogWindSpeed.setter
    def volumetricFogWindSpeed(self, arg: float, /) -> None: ...

    @property
    def volumetricFogTurbulenceSpeed(self) -> float: ...

    @volumetricFogTurbulenceSpeed.setter
    def volumetricFogTurbulenceSpeed(self, arg: float, /) -> None: ...

    @property
    def volumetricLightingEnabled(self) -> bool: ...

    @volumetricLightingEnabled.setter
    def volumetricLightingEnabled(self, arg: bool, /) -> None: ...

    @property
    def volumetricLightStrength(self) -> float: ...

    @volumetricLightStrength.setter
    def volumetricLightStrength(self, arg: float, /) -> None: ...

    @property
    def volumetricLightDecay(self) -> float: ...

    @volumetricLightDecay.setter
    def volumetricLightDecay(self, arg: float, /) -> None: ...

    @property
    def volumetricLightSamples(self) -> int: ...

    @volumetricLightSamples.setter
    def volumetricLightSamples(self, arg: int, /) -> None: ...

    @property
    def gamma(self) -> float: ...

    @gamma.setter
    def gamma(self, arg: float, /) -> None: ...

    @property
    def highFidelityPbr(self) -> bool: ...

    @highFidelityPbr.setter
    def highFidelityPbr(self, arg: bool, /) -> None: ...

    @property
    def forceSimpleMaterialShading(self) -> bool: ...

    @forceSimpleMaterialShading.setter
    def forceSimpleMaterialShading(self, arg: bool, /) -> None: ...

    @property
    def pbrToneMapping(self) -> bool: ...

    @pbrToneMapping.setter
    def pbrToneMapping(self, arg: bool, /) -> None: ...

    @property
    def colorMode(self) -> ViewerColorMode: ...

    @colorMode.setter
    def colorMode(self, arg: ViewerColorMode, /) -> None: ...

    @property
    def pbrExposure(self) -> float: ...

    @pbrExposure.setter
    def pbrExposure(self, arg: float, /) -> None: ...

    @property
    def autoExposureEnabled(self) -> bool: ...

    @autoExposureEnabled.setter
    def autoExposureEnabled(self, arg: bool, /) -> None: ...

    @property
    def autoExposureKey(self) -> float: ...

    @autoExposureKey.setter
    def autoExposureKey(self, arg: float, /) -> None: ...

    @property
    def autoExposureSpeed(self) -> float: ...

    @autoExposureSpeed.setter
    def autoExposureSpeed(self, arg: float, /) -> None: ...

    @property
    def autoExposureMinFactor(self) -> float: ...

    @autoExposureMinFactor.setter
    def autoExposureMinFactor(self, arg: float, /) -> None: ...

    @property
    def autoExposureMaxFactor(self) -> float: ...

    @autoExposureMaxFactor.setter
    def autoExposureMaxFactor(self, arg: float, /) -> None: ...

    @property
    def viewerWhiteBalance(self) -> numpy.ndarray: ...

    @viewerWhiteBalance.setter
    def viewerWhiteBalance(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def viewerSaturation(self) -> float: ...

    @viewerSaturation.setter
    def viewerSaturation(self, arg: float, /) -> None: ...

    @property
    def viewerColorGradePreset(self) -> ColorGradePreset: ...

    @viewerColorGradePreset.setter
    def viewerColorGradePreset(self, arg: ColorGradePreset, /) -> None: ...

    @property
    def viewerColorGradeStrength(self) -> float: ...

    @viewerColorGradeStrength.setter
    def viewerColorGradeStrength(self, arg: float, /) -> None: ...

    @property
    def pbrEnvironmentMaxLod(self) -> float: ...

    @pbrEnvironmentMaxLod.setter
    def pbrEnvironmentMaxLod(self, arg: float, /) -> None: ...

    @property
    def pbrEnvironmentMipLodBias(self) -> float: ...

    @pbrEnvironmentMipLodBias.setter
    def pbrEnvironmentMipLodBias(self, arg: float, /) -> None: ...

    @property
    def pbrEnvironmentRotation(self) -> float: ...

    @pbrEnvironmentRotation.setter
    def pbrEnvironmentRotation(self, arg: float, /) -> None: ...

    @property
    def pbrEnvironmentIntensity(self) -> float: ...

    @pbrEnvironmentIntensity.setter
    def pbrEnvironmentIntensity(self, arg: float, /) -> None: ...

    @property
    def pbrEnvironmentLightingTint(self) -> numpy.ndarray: ...

    @pbrEnvironmentLightingTint.setter
    def pbrEnvironmentLightingTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def pbrEnvironmentSkyTint(self) -> numpy.ndarray: ...

    @pbrEnvironmentSkyTint.setter
    def pbrEnvironmentSkyTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def pbrEnvironmentHorizonTint(self) -> numpy.ndarray: ...

    @pbrEnvironmentHorizonTint.setter
    def pbrEnvironmentHorizonTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def pbrEnvironmentGroundTint(self) -> numpy.ndarray: ...

    @pbrEnvironmentGroundTint.setter
    def pbrEnvironmentGroundTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def pbrEnvironmentHorizonStrength(self) -> float: ...

    @pbrEnvironmentHorizonStrength.setter
    def pbrEnvironmentHorizonStrength(self, arg: float, /) -> None: ...

    @property
    def environmentBackgroundTint(self) -> numpy.ndarray: ...

    @environmentBackgroundTint.setter
    def environmentBackgroundTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def environmentBackgroundExposureScale(self) -> float: ...

    @environmentBackgroundExposureScale.setter
    def environmentBackgroundExposureScale(self, arg: float, /) -> None: ...

    @property
    def proceduralSkyBackgroundEnabled(self) -> bool: ...

    @proceduralSkyBackgroundEnabled.setter
    def proceduralSkyBackgroundEnabled(self, arg: bool, /) -> None: ...

    @property
    def proceduralSkySunStrength(self) -> float: ...

    @proceduralSkySunStrength.setter
    def proceduralSkySunStrength(self, arg: float, /) -> None: ...

    @property
    def proceduralSkySunSize(self) -> float: ...

    @proceduralSkySunSize.setter
    def proceduralSkySunSize(self, arg: float, /) -> None: ...

    @property
    def proceduralCloudLayerEnabled(self) -> bool: ...

    @proceduralCloudLayerEnabled.setter
    def proceduralCloudLayerEnabled(self, arg: bool, /) -> None: ...

    @property
    def proceduralCloudCoverage(self) -> float: ...

    @proceduralCloudCoverage.setter
    def proceduralCloudCoverage(self, arg: float, /) -> None: ...

    @property
    def proceduralCloudDensity(self) -> float: ...

    @proceduralCloudDensity.setter
    def proceduralCloudDensity(self, arg: float, /) -> None: ...

    @property
    def proceduralCloudScale(self) -> float: ...

    @proceduralCloudScale.setter
    def proceduralCloudScale(self, arg: float, /) -> None: ...

    @property
    def proceduralCloudSoftness(self) -> float: ...

    @proceduralCloudSoftness.setter
    def proceduralCloudSoftness(self, arg: float, /) -> None: ...

    @property
    def proceduralCloudOffset(self) -> numpy.ndarray: ...

    @proceduralCloudOffset.setter
    def proceduralCloudOffset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def proceduralCloudTint(self) -> numpy.ndarray: ...

    @proceduralCloudTint.setter
    def proceduralCloudTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def cloudQuality(self) -> CloudQuality: ...

    @cloudQuality.setter
    def cloudQuality(self, arg: CloudQuality, /) -> None: ...

    @property
    def cloudShadowProjectionEnabled(self) -> bool: ...

    @cloudShadowProjectionEnabled.setter
    def cloudShadowProjectionEnabled(self, arg: bool, /) -> None: ...

    @property
    def cloudShadowStrength(self) -> float: ...

    @cloudShadowStrength.setter
    def cloudShadowStrength(self, arg: float, /) -> None: ...

    @property
    def cloudShadowScale(self) -> float: ...

    @cloudShadowScale.setter
    def cloudShadowScale(self, arg: float, /) -> None: ...

    @property
    def pbrKeyLightIntensity(self) -> float: ...

    @pbrKeyLightIntensity.setter
    def pbrKeyLightIntensity(self, arg: float, /) -> None: ...

    @property
    def specularAntiAliasing(self) -> bool: ...

    @specularAntiAliasing.setter
    def specularAntiAliasing(self, arg: bool, /) -> None: ...

    @property
    def specularAntiAliasingStrength(self) -> float: ...

    @specularAntiAliasingStrength.setter
    def specularAntiAliasingStrength(self, arg: float, /) -> None: ...

    @property
    def reflectiveGround(self) -> bool: ...

    @reflectiveGround.setter
    def reflectiveGround(self, arg: bool, /) -> None: ...

    @property
    def reflectiveGroundRoughness(self) -> float: ...

    @reflectiveGroundRoughness.setter
    def reflectiveGroundRoughness(self, arg: float, /) -> None: ...

    @property
    def reflectiveGroundMetallic(self) -> float: ...

    @reflectiveGroundMetallic.setter
    def reflectiveGroundMetallic(self, arg: float, /) -> None: ...

    @property
    def fxaaEnabled(self) -> bool: ...

    @fxaaEnabled.setter
    def fxaaEnabled(self, arg: bool, /) -> None: ...

    @property
    def viewerVignetteStrength(self) -> float: ...

    @viewerVignetteStrength.setter
    def viewerVignetteStrength(self, arg: float, /) -> None: ...

    @property
    def viewerChromaticAberrationStrength(self) -> float: ...

    @viewerChromaticAberrationStrength.setter
    def viewerChromaticAberrationStrength(self, arg: float, /) -> None: ...

    @property
    def viewerFilmGrainStrength(self) -> float: ...

    @viewerFilmGrainStrength.setter
    def viewerFilmGrainStrength(self, arg: float, /) -> None: ...

    @property
    def aerialPerspectiveEnabled(self) -> bool: ...

    @aerialPerspectiveEnabled.setter
    def aerialPerspectiveEnabled(self, arg: bool, /) -> None: ...

    @property
    def aerialPerspectiveStrength(self) -> float: ...

    @aerialPerspectiveStrength.setter
    def aerialPerspectiveStrength(self, arg: float, /) -> None: ...

    @property
    def detailNormalEnabled(self) -> bool: ...

    @detailNormalEnabled.setter
    def detailNormalEnabled(self, arg: bool, /) -> None: ...

    @property
    def detailNormalScale(self) -> float: ...

    @detailNormalScale.setter
    def detailNormalScale(self, arg: float, /) -> None: ...

    @property
    def detailNormalStrength(self) -> float: ...

    @detailNormalStrength.setter
    def detailNormalStrength(self, arg: float, /) -> None: ...

    @property
    def viewerSubsurfaceWrap(self) -> float: ...

    @viewerSubsurfaceWrap.setter
    def viewerSubsurfaceWrap(self, arg: float, /) -> None: ...

    @property
    def viewerSubsurfaceTint(self) -> numpy.ndarray: ...

    @viewerSubsurfaceTint.setter
    def viewerSubsurfaceTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lightShaftsEnabled(self) -> bool: ...

    @lightShaftsEnabled.setter
    def lightShaftsEnabled(self, arg: bool, /) -> None: ...

    @property
    def lightShaftsStrength(self) -> float: ...

    @lightShaftsStrength.setter
    def lightShaftsStrength(self, arg: float, /) -> None: ...

    @property
    def lightShaftsDecay(self) -> float: ...

    @lightShaftsDecay.setter
    def lightShaftsDecay(self, arg: float, /) -> None: ...

    @property
    def lightShaftsDensity(self) -> float: ...

    @lightShaftsDensity.setter
    def lightShaftsDensity(self, arg: float, /) -> None: ...

    @property
    def lensFlareGhostStrength(self) -> float: ...

    @lensFlareGhostStrength.setter
    def lensFlareGhostStrength(self, arg: float, /) -> None: ...

    @property
    def lensFlareStreakStrength(self) -> float: ...

    @lensFlareStreakStrength.setter
    def lensFlareStreakStrength(self, arg: float, /) -> None: ...

    @property
    def lensFlareTint(self) -> numpy.ndarray: ...

    @lensFlareTint.setter
    def lensFlareTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def contactShadowsEnabled(self) -> bool: ...

    @contactShadowsEnabled.setter
    def contactShadowsEnabled(self, arg: bool, /) -> None: ...

    @property
    def contactShadowsLength(self) -> float: ...

    @contactShadowsLength.setter
    def contactShadowsLength(self, arg: float, /) -> None: ...

    @property
    def contactShadowsStrength(self) -> float: ...

    @contactShadowsStrength.setter
    def contactShadowsStrength(self, arg: float, /) -> None: ...

    @property
    def contactShadowsThickness(self) -> float: ...

    @contactShadowsThickness.setter
    def contactShadowsThickness(self, arg: float, /) -> None: ...

    @property
    def motionBlurEnabled(self) -> bool: ...

    @motionBlurEnabled.setter
    def motionBlurEnabled(self, arg: bool, /) -> None: ...

    @property
    def motionBlurDirection(self) -> numpy.ndarray: ...

    @motionBlurDirection.setter
    def motionBlurDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def motionBlurStrength(self) -> float: ...

    @motionBlurStrength.setter
    def motionBlurStrength(self, arg: float, /) -> None: ...

    @property
    def ssrEnabled(self) -> bool: ...

    @ssrEnabled.setter
    def ssrEnabled(self, arg: bool, /) -> None: ...

    @property
    def ssrStrength(self) -> float: ...

    @ssrStrength.setter
    def ssrStrength(self, arg: float, /) -> None: ...

    @property
    def ssrSteps(self) -> int: ...

    @ssrSteps.setter
    def ssrSteps(self, arg: int, /) -> None: ...

    @property
    def ssrMaxDistance(self) -> float: ...

    @ssrMaxDistance.setter
    def ssrMaxDistance(self, arg: float, /) -> None: ...

    @property
    def ssrThickness(self) -> float: ...

    @ssrThickness.setter
    def ssrThickness(self, arg: float, /) -> None: ...

    @property
    def lensDistortionEnabled(self) -> bool: ...

    @lensDistortionEnabled.setter
    def lensDistortionEnabled(self, arg: bool, /) -> None: ...

    @property
    def lensDistortionStrength(self) -> float: ...

    @lensDistortionStrength.setter
    def lensDistortionStrength(self, arg: float, /) -> None: ...

    @property
    def lensDistortionCenter(self) -> numpy.ndarray: ...

    @lensDistortionCenter.setter
    def lensDistortionCenter(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def pixelateEnabled(self) -> bool: ...

    @pixelateEnabled.setter
    def pixelateEnabled(self, arg: bool, /) -> None: ...

    @property
    def pixelateBlockSize(self) -> int: ...

    @pixelateBlockSize.setter
    def pixelateBlockSize(self, arg: int, /) -> None: ...

    @property
    def nightVisionEnabled(self) -> bool: ...

    @nightVisionEnabled.setter
    def nightVisionEnabled(self, arg: bool, /) -> None: ...

    @property
    def nightVisionStrength(self) -> float: ...

    @nightVisionStrength.setter
    def nightVisionStrength(self, arg: float, /) -> None: ...

    @property
    def nightVisionGain(self) -> float: ...

    @nightVisionGain.setter
    def nightVisionGain(self, arg: float, /) -> None: ...

    @property
    def ditherEnabled(self) -> bool: ...

    @ditherEnabled.setter
    def ditherEnabled(self, arg: bool, /) -> None: ...

    @property
    def ditherStrength(self) -> float: ...

    @ditherStrength.setter
    def ditherStrength(self, arg: float, /) -> None: ...

    @property
    def thermalEnabled(self) -> bool: ...

    @thermalEnabled.setter
    def thermalEnabled(self, arg: bool, /) -> None: ...

    @property
    def thermalStrength(self) -> float: ...

    @thermalStrength.setter
    def thermalStrength(self, arg: float, /) -> None: ...

    @property
    def thermalGain(self) -> float: ...

    @thermalGain.setter
    def thermalGain(self, arg: float, /) -> None: ...

    @property
    def scanlineEnabled(self) -> bool: ...

    @scanlineEnabled.setter
    def scanlineEnabled(self, arg: bool, /) -> None: ...

    @property
    def scanlineStrength(self) -> float: ...

    @scanlineStrength.setter
    def scanlineStrength(self, arg: float, /) -> None: ...

    @property
    def scanlineFrequency(self) -> float: ...

    @scanlineFrequency.setter
    def scanlineFrequency(self, arg: float, /) -> None: ...

    @property
    def letterboxEnabled(self) -> bool: ...

    @letterboxEnabled.setter
    def letterboxEnabled(self, arg: bool, /) -> None: ...

    @property
    def letterboxAspect(self) -> float: ...

    @letterboxAspect.setter
    def letterboxAspect(self, arg: float, /) -> None: ...

    @property
    def letterboxColor(self) -> numpy.ndarray: ...

    @letterboxColor.setter
    def letterboxColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def underwaterEnabled(self) -> bool: ...

    @underwaterEnabled.setter
    def underwaterEnabled(self, arg: bool, /) -> None: ...

    @property
    def underwaterTint(self) -> numpy.ndarray: ...

    @underwaterTint.setter
    def underwaterTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def underwaterCausticStrength(self) -> float: ...

    @underwaterCausticStrength.setter
    def underwaterCausticStrength(self, arg: float, /) -> None: ...

    @property
    def underwaterCausticScale(self) -> float: ...

    @underwaterCausticScale.setter
    def underwaterCausticScale(self, arg: float, /) -> None: ...

    @property
    def underwaterDepthExtinction(self) -> float: ...

    @underwaterDepthExtinction.setter
    def underwaterDepthExtinction(self, arg: float, /) -> None: ...

    @property
    def heatHazeEnabled(self) -> bool: ...

    @heatHazeEnabled.setter
    def heatHazeEnabled(self, arg: bool, /) -> None: ...

    @property
    def heatHazeStrength(self) -> float: ...

    @heatHazeStrength.setter
    def heatHazeStrength(self, arg: float, /) -> None: ...

    @property
    def heatHazeFrequency(self) -> float: ...

    @heatHazeFrequency.setter
    def heatHazeFrequency(self, arg: float, /) -> None: ...

    @property
    def heatHazeSpeed(self) -> float: ...

    @heatHazeSpeed.setter
    def heatHazeSpeed(self, arg: float, /) -> None: ...

    @property
    def heatHazeMaxY(self) -> float: ...

    @heatHazeMaxY.setter
    def heatHazeMaxY(self, arg: float, /) -> None: ...

    @property
    def zoomBlurEnabled(self) -> bool: ...

    @zoomBlurEnabled.setter
    def zoomBlurEnabled(self, arg: bool, /) -> None: ...

    @property
    def zoomBlurStrength(self) -> float: ...

    @zoomBlurStrength.setter
    def zoomBlurStrength(self, arg: float, /) -> None: ...

    @property
    def zoomBlurCenter(self) -> numpy.ndarray: ...

    @zoomBlurCenter.setter
    def zoomBlurCenter(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def zoomBlurInnerRadius(self) -> float: ...

    @zoomBlurInnerRadius.setter
    def zoomBlurInnerRadius(self, arg: float, /) -> None: ...

    @property
    def posterizeEnabled(self) -> bool: ...

    @posterizeEnabled.setter
    def posterizeEnabled(self, arg: bool, /) -> None: ...

    @property
    def posterizeLevels(self) -> int: ...

    @posterizeLevels.setter
    def posterizeLevels(self, arg: int, /) -> None: ...

    @property
    def starburstEnabled(self) -> bool: ...

    @starburstEnabled.setter
    def starburstEnabled(self, arg: bool, /) -> None: ...

    @property
    def starburstStrength(self) -> float: ...

    @starburstStrength.setter
    def starburstStrength(self, arg: float, /) -> None: ...

    @property
    def starburstSpikes(self) -> int: ...

    @starburstSpikes.setter
    def starburstSpikes(self, arg: int, /) -> None: ...

    @property
    def starburstAngleOffset(self) -> float: ...

    @starburstAngleOffset.setter
    def starburstAngleOffset(self, arg: float, /) -> None: ...

    @property
    def starburstRadius(self) -> float: ...

    @starburstRadius.setter
    def starburstRadius(self, arg: float, /) -> None: ...

    @property
    def starburstTint(self) -> numpy.ndarray: ...

    @starburstTint.setter
    def starburstTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def bloomEnabled(self) -> bool: ...

    @bloomEnabled.setter
    def bloomEnabled(self, arg: bool, /) -> None: ...

    @property
    def bloomThreshold(self) -> float: ...

    @bloomThreshold.setter
    def bloomThreshold(self, arg: float, /) -> None: ...

    @property
    def bloomStrength(self) -> float: ...

    @bloomStrength.setter
    def bloomStrength(self, arg: float, /) -> None: ...

    @property
    def bloomRadius(self) -> float: ...

    @bloomRadius.setter
    def bloomRadius(self, arg: float, /) -> None: ...

    @property
    def bloomKnee(self) -> float: ...

    @bloomKnee.setter
    def bloomKnee(self, arg: float, /) -> None: ...

    @property
    def bloomQuality(self) -> int: ...

    @bloomQuality.setter
    def bloomQuality(self, arg: int, /) -> None: ...

    @property
    def bloomSourceClamp(self) -> float: ...

    @bloomSourceClamp.setter
    def bloomSourceClamp(self, arg: float, /) -> None: ...

    @property
    def bloomAnamorphic(self) -> float: ...

    @bloomAnamorphic.setter
    def bloomAnamorphic(self, arg: float, /) -> None: ...

    @property
    def bloomDirtStrength(self) -> float: ...

    @bloomDirtStrength.setter
    def bloomDirtStrength(self, arg: float, /) -> None: ...

    @property
    def bloomDirtScale(self) -> float: ...

    @bloomDirtScale.setter
    def bloomDirtScale(self, arg: float, /) -> None: ...

    @property
    def bloomDirtTexture(self) -> int: ...

    @bloomDirtTexture.setter
    def bloomDirtTexture(self, arg: int, /) -> None: ...

    @property
    def bloomBlendMode(self) -> GlowBlendMode: ...

    @bloomBlendMode.setter
    def bloomBlendMode(self, arg: GlowBlendMode, /) -> None: ...

    @property
    def bloomMix(self) -> float: ...

    @bloomMix.setter
    def bloomMix(self, arg: float, /) -> None: ...

    @property
    def postProcessDebugMode(self) -> PostProcessDebugMode: ...

    @postProcessDebugMode.setter
    def postProcessDebugMode(self, arg: PostProcessDebugMode, /) -> None: ...

    @property
    def screenSpaceAoEnabled(self) -> bool: ...

    @screenSpaceAoEnabled.setter
    def screenSpaceAoEnabled(self, arg: bool, /) -> None: ...

    @property
    def screenSpaceAoRadius(self) -> float: ...

    @screenSpaceAoRadius.setter
    def screenSpaceAoRadius(self, arg: float, /) -> None: ...

    @property
    def screenSpaceAoStrength(self) -> float: ...

    @screenSpaceAoStrength.setter
    def screenSpaceAoStrength(self, arg: float, /) -> None: ...

    @property
    def screenSpaceAoBias(self) -> float: ...

    @screenSpaceAoBias.setter
    def screenSpaceAoBias(self, arg: float, /) -> None: ...

    @property
    def screenSpaceAoSamples(self) -> int: ...

    @screenSpaceAoSamples.setter
    def screenSpaceAoSamples(self, arg: int, /) -> None: ...

    @property
    def screenSpaceAoFalloff(self) -> float: ...

    @screenSpaceAoFalloff.setter
    def screenSpaceAoFalloff(self, arg: float, /) -> None: ...

    @property
    def screenSpaceAoRadiusWorldSpace(self) -> bool: ...

    @screenSpaceAoRadiusWorldSpace.setter
    def screenSpaceAoRadiusWorldSpace(self, arg: bool, /) -> None: ...

    @property
    def screenSpaceAoMaxPixelRadius(self) -> float: ...

    @screenSpaceAoMaxPixelRadius.setter
    def screenSpaceAoMaxPixelRadius(self, arg: float, /) -> None: ...

    @property
    def screenSpaceAoGeometryAware(self) -> bool: ...

    @screenSpaceAoGeometryAware.setter
    def screenSpaceAoGeometryAware(self, arg: bool, /) -> None: ...

    @property
    def contactAoRadius(self) -> float: ...

    @contactAoRadius.setter
    def contactAoRadius(self, arg: float, /) -> None: ...

    @property
    def contactAoStrength(self) -> float: ...

    @contactAoStrength.setter
    def contactAoStrength(self, arg: float, /) -> None: ...

    @property
    def contactAoSamples(self) -> int: ...

    @contactAoSamples.setter
    def contactAoSamples(self, arg: int, /) -> None: ...

    @property
    def contactAoFalloff(self) -> float: ...

    @contactAoFalloff.setter
    def contactAoFalloff(self, arg: float, /) -> None: ...

    @property
    def screenSpaceAoDenoiseEnabled(self) -> bool: ...

    @screenSpaceAoDenoiseEnabled.setter
    def screenSpaceAoDenoiseEnabled(self, arg: bool, /) -> None: ...

    @property
    def screenSpaceAoDenoiseStrength(self) -> float: ...

    @screenSpaceAoDenoiseStrength.setter
    def screenSpaceAoDenoiseStrength(self, arg: float, /) -> None: ...

    @property
    def screenSpaceAoDenoiseRadius(self) -> int: ...

    @screenSpaceAoDenoiseRadius.setter
    def screenSpaceAoDenoiseRadius(self, arg: int, /) -> None: ...

    @property
    def screenSpaceAoDenoiseDepthSigma(self) -> float: ...

    @screenSpaceAoDenoiseDepthSigma.setter
    def screenSpaceAoDenoiseDepthSigma(self, arg: float, /) -> None: ...

    @property
    def screenSpaceIndirectLightingEnabled(self) -> bool: ...

    @screenSpaceIndirectLightingEnabled.setter
    def screenSpaceIndirectLightingEnabled(self, arg: bool, /) -> None: ...

    @property
    def screenSpaceIndirectLightingRadius(self) -> float: ...

    @screenSpaceIndirectLightingRadius.setter
    def screenSpaceIndirectLightingRadius(self, arg: float, /) -> None: ...

    @property
    def screenSpaceIndirectLightingStrength(self) -> float: ...

    @screenSpaceIndirectLightingStrength.setter
    def screenSpaceIndirectLightingStrength(self, arg: float, /) -> None: ...

    @property
    def screenSpaceIndirectLightingSamples(self) -> int: ...

    @screenSpaceIndirectLightingSamples.setter
    def screenSpaceIndirectLightingSamples(self, arg: int, /) -> None: ...

    @property
    def screenSpaceIndirectLightingFalloff(self) -> float: ...

    @screenSpaceIndirectLightingFalloff.setter
    def screenSpaceIndirectLightingFalloff(self, arg: float, /) -> None: ...

    @property
    def screenSpaceIndirectLightingNormalRejection(self) -> float: ...

    @screenSpaceIndirectLightingNormalRejection.setter
    def screenSpaceIndirectLightingNormalRejection(self, arg: float, /) -> None: ...

    @property
    def screenSpaceIndirectLightingSaturation(self) -> float: ...

    @screenSpaceIndirectLightingSaturation.setter
    def screenSpaceIndirectLightingSaturation(self, arg: float, /) -> None: ...

    @property
    def opaqueDepthPrepass(self) -> bool: ...

    @opaqueDepthPrepass.setter
    def opaqueDepthPrepass(self, arg: bool, /) -> None: ...

    @property
    def frontToBackOpaqueInstancedOrdering(self) -> bool: ...

    @frontToBackOpaqueInstancedOrdering.setter
    def frontToBackOpaqueInstancedOrdering(self, arg: bool, /) -> None: ...

    @property
    def foliageAlphaDepthPrepass(self) -> bool: ...

    @foliageAlphaDepthPrepass.setter
    def foliageAlphaDepthPrepass(self, arg: bool, /) -> None: ...

    @property
    def depthOfFieldEnabled(self) -> bool: ...

    @depthOfFieldEnabled.setter
    def depthOfFieldEnabled(self, arg: bool, /) -> None: ...

    @property
    def depthOfFieldFocusDistance(self) -> float: ...

    @depthOfFieldFocusDistance.setter
    def depthOfFieldFocusDistance(self, arg: float, /) -> None: ...

    @property
    def depthOfFieldFocusRange(self) -> float: ...

    @depthOfFieldFocusRange.setter
    def depthOfFieldFocusRange(self, arg: float, /) -> None: ...

    @property
    def depthOfFieldMaxRadius(self) -> float: ...

    @depthOfFieldMaxRadius.setter
    def depthOfFieldMaxRadius(self, arg: float, /) -> None: ...

    @property
    def textureAnisotropy(self) -> float: ...

    @textureAnisotropy.setter
    def textureAnisotropy(self, arg: float, /) -> None: ...

    @property
    def textureMipLodBias(self) -> float: ...

    @textureMipLodBias.setter
    def textureMipLodBias(self, arg: float, /) -> None: ...

    @property
    def viewerMsaaSamples(self) -> int: ...

    @viewerMsaaSamples.setter
    def viewerMsaaSamples(self, arg: int, /) -> None: ...

    @property
    def alphaToCoverage(self) -> bool: ...

    @alphaToCoverage.setter
    def alphaToCoverage(self, arg: bool, /) -> None: ...

    @property
    def temporalAaEnabled(self) -> bool: ...

    @temporalAaEnabled.setter
    def temporalAaEnabled(self, arg: bool, /) -> None: ...

    @property
    def temporalAaBlend(self) -> float: ...

    @temporalAaBlend.setter
    def temporalAaBlend(self, arg: float, /) -> None: ...

    @property
    def temporalAaJitterScale(self) -> float: ...

    @temporalAaJitterScale.setter
    def temporalAaJitterScale(self, arg: float, /) -> None: ...

    @property
    def temporalAaMethod(self) -> TemporalAaMethod: ...

    @temporalAaMethod.setter
    def temporalAaMethod(self, arg: TemporalAaMethod, /) -> None: ...

    @property
    def temporalAaCurrentWeight(self) -> float: ...

    @temporalAaCurrentWeight.setter
    def temporalAaCurrentWeight(self, arg: float, /) -> None: ...

    @property
    def temporalAaSharpness(self) -> float: ...

    @temporalAaSharpness.setter
    def temporalAaSharpness(self, arg: float, /) -> None: ...

    @property
    def viewerRenderScale(self) -> float: ...

    @viewerRenderScale.setter
    def viewerRenderScale(self, arg: float, /) -> None: ...

    @property
    def viewerUpscaleFilter(self) -> ViewerUpscaleFilter: ...

    @viewerUpscaleFilter.setter
    def viewerUpscaleFilter(self, arg: ViewerUpscaleFilter, /) -> None: ...

    @property
    def shadowsEnabled(self) -> bool: ...

    @shadowsEnabled.setter
    def shadowsEnabled(self, arg: bool, /) -> None: ...

    @property
    def shadowedLightBudget(self) -> int: ...

    @shadowedLightBudget.setter
    def shadowedLightBudget(self, arg: int, /) -> None: ...

    @property
    def pointShadowPolicy(self) -> PointShadowPolicy: ...

    @pointShadowPolicy.setter
    def pointShadowPolicy(self, arg: PointShadowPolicy, /) -> None: ...

    @property
    def maxPointShadowLights(self) -> int: ...

    @maxPointShadowLights.setter
    def maxPointShadowLights(self, arg: int, /) -> None: ...

    @property
    def additionalShadowResolutionScale(self) -> float: ...

    @additionalShadowResolutionScale.setter
    def additionalShadowResolutionScale(self, arg: float, /) -> None: ...

    @property
    def pointShadowResolutionScale(self) -> float: ...

    @pointShadowResolutionScale.setter
    def pointShadowResolutionScale(self, arg: float, /) -> None: ...

    @property
    def minAdditionalShadowResolution(self) -> int: ...

    @minAdditionalShadowResolution.setter
    def minAdditionalShadowResolution(self, arg: int, /) -> None: ...

    @property
    def shadowSlopeBiasScale(self) -> float: ...

    @shadowSlopeBiasScale.setter
    def shadowSlopeBiasScale(self, arg: float, /) -> None: ...

    @property
    def shadowNormalOffset(self) -> float: ...

    @shadowNormalOffset.setter
    def shadowNormalOffset(self, arg: float, /) -> None: ...

    @property
    def directionalShadowCascadeCount(self) -> int: ...

    @directionalShadowCascadeCount.setter
    def directionalShadowCascadeCount(self, arg: int, /) -> None: ...

    @property
    def directionalShadowCascadeLambda(self) -> float: ...

    @directionalShadowCascadeLambda.setter
    def directionalShadowCascadeLambda(self, arg: float, /) -> None: ...

    @property
    def directionalShadowCascadeMaxDistance(self) -> float: ...

    @directionalShadowCascadeMaxDistance.setter
    def directionalShadowCascadeMaxDistance(self, arg: float, /) -> None: ...

    @property
    def directionalShadowCascadeSplitOverrides(self) -> list[float]: ...

    @directionalShadowCascadeSplitOverrides.setter
    def directionalShadowCascadeSplitOverrides(self, arg: Sequence[float], /) -> None: ...

    @property
    def additionalShadowBiasScale(self) -> float: ...

    @additionalShadowBiasScale.setter
    def additionalShadowBiasScale(self, arg: float, /) -> None: ...

    @property
    def additionalShadowStrengthScale(self) -> float: ...

    @additionalShadowStrengthScale.setter
    def additionalShadowStrengthScale(self, arg: float, /) -> None: ...

    @property
    def additionalShadowPcfRadiusScale(self) -> float: ...

    @additionalShadowPcfRadiusScale.setter
    def additionalShadowPcfRadiusScale(self, arg: float, /) -> None: ...

    @property
    def additionalShadowNormalOffsetScale(self) -> float: ...

    @additionalShadowNormalOffsetScale.setter
    def additionalShadowNormalOffsetScale(self, arg: float, /) -> None: ...

    @property
    def additionalShadowAtlasPlanningEnabled(self) -> bool: ...

    @additionalShadowAtlasPlanningEnabled.setter
    def additionalShadowAtlasPlanningEnabled(self, arg: bool, /) -> None: ...

    @property
    def additionalShadowAtlasResolution(self) -> int: ...

    @additionalShadowAtlasResolution.setter
    def additionalShadowAtlasResolution(self, arg: int, /) -> None: ...

    @property
    def additionalShadowAtlasPadding(self) -> int: ...

    @additionalShadowAtlasPadding.setter
    def additionalShadowAtlasPadding(self, arg: int, /) -> None: ...

    @property
    def pointShadowBiasScale(self) -> float: ...

    @pointShadowBiasScale.setter
    def pointShadowBiasScale(self, arg: float, /) -> None: ...

    @property
    def pointShadowFar(self) -> float: ...

    @pointShadowFar.setter
    def pointShadowFar(self, arg: float, /) -> None: ...

    @property
    def updateShadowsEveryFrame(self) -> bool: ...

    @updateShadowsEveryFrame.setter
    def updateShadowsEveryFrame(self, arg: bool, /) -> None: ...

    @property
    def maxAdditionalLightsPerFrame(self) -> int: ...

    @maxAdditionalLightsPerFrame.setter
    def maxAdditionalLightsPerFrame(self, arg: int, /) -> None: ...

    @property
    def minAdditionalLightInfluence(self) -> float: ...

    @minAdditionalLightInfluence.setter
    def minAdditionalLightInfluence(self, arg: float, /) -> None: ...

    @property
    def additionalLightContributionThreshold(self) -> float: ...

    @additionalLightContributionThreshold.setter
    def additionalLightContributionThreshold(self, arg: float, /) -> None: ...

    @property
    def maxReflectionProbesPerSelection(self) -> int: ...

    @maxReflectionProbesPerSelection.setter
    def maxReflectionProbesPerSelection(self, arg: int, /) -> None: ...

    @property
    def minReflectionProbeWeight(self) -> float: ...

    @minReflectionProbeWeight.setter
    def minReflectionProbeWeight(self, arg: float, /) -> None: ...

    @property
    def reflectionProbeDebugOverlayEnabled(self) -> bool: ...

    @reflectionProbeDebugOverlayEnabled.setter
    def reflectionProbeDebugOverlayEnabled(self, arg: bool, /) -> None: ...

    @property
    def reflectionProbeFilteringEnabled(self) -> bool: ...

    @reflectionProbeFilteringEnabled.setter
    def reflectionProbeFilteringEnabled(self, arg: bool, /) -> None: ...

    @property
    def reflectionProbeIrradianceResolution(self) -> int: ...

    @reflectionProbeIrradianceResolution.setter
    def reflectionProbeIrradianceResolution(self, arg: int, /) -> None: ...

    @property
    def reflectionProbeIrradianceSamples(self) -> int: ...

    @reflectionProbeIrradianceSamples.setter
    def reflectionProbeIrradianceSamples(self, arg: int, /) -> None: ...

    @property
    def reflectionProbePrefilteredResolution(self) -> int: ...

    @reflectionProbePrefilteredResolution.setter
    def reflectionProbePrefilteredResolution(self, arg: int, /) -> None: ...

    @property
    def reflectionProbePrefilteredMipLevels(self) -> int: ...

    @reflectionProbePrefilteredMipLevels.setter
    def reflectionProbePrefilteredMipLevels(self, arg: int, /) -> None: ...

    @property
    def reflectionProbePrefilteredSamples(self) -> int: ...

    @reflectionProbePrefilteredSamples.setter
    def reflectionProbePrefilteredSamples(self, arg: int, /) -> None: ...

    @property
    def reflectionProbeBrdfLutSize(self) -> int: ...

    @reflectionProbeBrdfLutSize.setter
    def reflectionProbeBrdfLutSize(self, arg: int, /) -> None: ...

    @property
    def reflectionProbeBrdfLutSamples(self) -> int: ...

    @reflectionProbeBrdfLutSamples.setter
    def reflectionProbeBrdfLutSamples(self, arg: int, /) -> None: ...

    @property
    def localDiffuseIrradianceIntensity(self) -> float: ...

    @localDiffuseIrradianceIntensity.setter
    def localDiffuseIrradianceIntensity(self, arg: float, /) -> None: ...

    @property
    def localDiffuseIrradianceSaturation(self) -> float: ...

    @localDiffuseIrradianceSaturation.setter
    def localDiffuseIrradianceSaturation(self, arg: float, /) -> None: ...

    @property
    def autoSelectImportedShadowLight(self) -> bool: ...

    @autoSelectImportedShadowLight.setter
    def autoSelectImportedShadowLight(self, arg: bool, /) -> None: ...

    @property
    def sortTransparentInstances(self) -> bool: ...

    @sortTransparentInstances.setter
    def sortTransparentInstances(self, arg: bool, /) -> None: ...

    @property
    def weightedBlendedOit(self) -> bool: ...

    @weightedBlendedOit.setter
    def weightedBlendedOit(self, arg: bool, /) -> None: ...

    @property
    def transparentDepthPrepass(self) -> bool: ...

    @transparentDepthPrepass.setter
    def transparentDepthPrepass(self, arg: bool, /) -> None: ...

    @property
    def transparentShadowDithering(self) -> bool: ...

    @transparentShadowDithering.setter
    def transparentShadowDithering(self, arg: bool, /) -> None: ...

    @property
    def transparentShadowOpacityScale(self) -> float: ...

    @transparentShadowOpacityScale.setter
    def transparentShadowOpacityScale(self, arg: float, /) -> None: ...

    @property
    def screenSpaceRefraction(self) -> bool: ...

    @screenSpaceRefraction.setter
    def screenSpaceRefraction(self, arg: bool, /) -> None: ...

    @property
    def screenSpaceRefractionStrength(self) -> float: ...

    @screenSpaceRefractionStrength.setter
    def screenSpaceRefractionStrength(self, arg: float, /) -> None: ...

    @property
    def screenSpaceRefractionMaxPixels(self) -> float: ...

    @screenSpaceRefractionMaxPixels.setter
    def screenSpaceRefractionMaxPixels(self, arg: float, /) -> None: ...

    @property
    def geometryRefraction(self) -> bool: ...

    @geometryRefraction.setter
    def geometryRefraction(self, arg: bool, /) -> None: ...

    @property
    def geometryRefractionBackend(self) -> GeometryRefractionBackend: ...

    @geometryRefractionBackend.setter
    def geometryRefractionBackend(self, arg: GeometryRefractionBackend, /) -> None: ...

    @property
    def geometryRefractionSamples(self) -> int: ...

    @geometryRefractionSamples.setter
    def geometryRefractionSamples(self, arg: int, /) -> None: ...

    @property
    def geometryRefractionMaxBounces(self) -> int: ...

    @geometryRefractionMaxBounces.setter
    def geometryRefractionMaxBounces(self, arg: int, /) -> None: ...

    @property
    def geometryRefractionProgressive(self) -> bool: ...

    @geometryRefractionProgressive.setter
    def geometryRefractionProgressive(self, arg: bool, /) -> None: ...

    @property
    def geometryRefractionMaxSamples(self) -> int: ...

    @geometryRefractionMaxSamples.setter
    def geometryRefractionMaxSamples(self, arg: int, /) -> None: ...

    @property
    def geometryRefractionLowDiscrepancy(self) -> bool: ...

    @geometryRefractionLowDiscrepancy.setter
    def geometryRefractionLowDiscrepancy(self, arg: bool, /) -> None: ...

    @property
    def geometryRefractionAdaptive(self) -> bool: ...

    @geometryRefractionAdaptive.setter
    def geometryRefractionAdaptive(self, arg: bool, /) -> None: ...

    @property
    def geometryRefractionMinSamples(self) -> int: ...

    @geometryRefractionMinSamples.setter
    def geometryRefractionMinSamples(self, arg: int, /) -> None: ...

    @property
    def geometryRefractionRelativeError(self) -> float: ...

    @geometryRefractionRelativeError.setter
    def geometryRefractionRelativeError(self, arg: float, /) -> None: ...

    @property
    def geometryRefractionAbsoluteError(self) -> float: ...

    @geometryRefractionAbsoluteError.setter
    def geometryRefractionAbsoluteError(self, arg: float, /) -> None: ...

    @property
    def geometryRefractionSeed(self) -> int: ...

    @geometryRefractionSeed.setter
    def geometryRefractionSeed(self, arg: int, /) -> None: ...

    @property
    def geometryRefractionRevision(self) -> int: ...

    @geometryRefractionRevision.setter
    def geometryRefractionRevision(self, arg: int, /) -> None: ...

    @property
    def transparentFogAlphaWeighted(self) -> bool: ...

    @transparentFogAlphaWeighted.setter
    def transparentFogAlphaWeighted(self, arg: bool, /) -> None: ...

    @property
    def transparentFogStrength(self) -> float: ...

    @transparentFogStrength.setter
    def transparentFogStrength(self, arg: float, /) -> None: ...

    @property
    def waterEnabled(self) -> bool: ...

    @waterEnabled.setter
    def waterEnabled(self, arg: bool, /) -> None: ...

    @property
    def waterTileResolution(self) -> int: ...

    @waterTileResolution.setter
    def waterTileResolution(self, arg: int, /) -> None: ...

    @property
    def waterLodDistanceScale(self) -> float: ...

    @waterLodDistanceScale.setter
    def waterLodDistanceScale(self, arg: float, /) -> None: ...

    @property
    def waterScreenSpaceReflections(self) -> bool: ...

    @waterScreenSpaceReflections.setter
    def waterScreenSpaceReflections(self, arg: bool, /) -> None: ...

    @property
    def waterReflectionSteps(self) -> int: ...

    @waterReflectionSteps.setter
    def waterReflectionSteps(self, arg: int, /) -> None: ...

    @property
    def waterReflectionMaxDistance(self) -> float: ...

    @waterReflectionMaxDistance.setter
    def waterReflectionMaxDistance(self, arg: float, /) -> None: ...

    @property
    def waterCaustics(self) -> bool: ...

    @waterCaustics.setter
    def waterCaustics(self, arg: bool, /) -> None: ...

    @property
    def waterUnderwaterView(self) -> bool: ...

    @waterUnderwaterView.setter
    def waterUnderwaterView(self, arg: bool, /) -> None: ...

    @property
    def waterSkyCaptureResolution(self) -> int: ...

    @waterSkyCaptureResolution.setter
    def waterSkyCaptureResolution(self, arg: int, /) -> None: ...

    @property
    def weatherWetMaterialEnabled(self) -> bool: ...

    @weatherWetMaterialEnabled.setter
    def weatherWetMaterialEnabled(self, arg: bool, /) -> None: ...

    @property
    def weatherWetness(self) -> float: ...

    @weatherWetness.setter
    def weatherWetness(self, arg: float, /) -> None: ...

    @property
    def weatherPuddleStrength(self) -> float: ...

    @weatherPuddleStrength.setter
    def weatherPuddleStrength(self, arg: float, /) -> None: ...

    @property
    def weatherRainRippleStrength(self) -> float: ...

    @weatherRainRippleStrength.setter
    def weatherRainRippleStrength(self, arg: float, /) -> None: ...

    @property
    def weatherRainRippleScale(self) -> float: ...

    @weatherRainRippleScale.setter
    def weatherRainRippleScale(self, arg: float, /) -> None: ...

    @property
    def weatherRainRipplePhase(self) -> float: ...

    @weatherRainRipplePhase.setter
    def weatherRainRipplePhase(self, arg: float, /) -> None: ...

    @property
    def weatherWetAlbedoDarkening(self) -> float: ...

    @weatherWetAlbedoDarkening.setter
    def weatherWetAlbedoDarkening(self, arg: float, /) -> None: ...

    @property
    def weatherWetRoughnessScale(self) -> float: ...

    @weatherWetRoughnessScale.setter
    def weatherWetRoughnessScale(self, arg: float, /) -> None: ...

    @property
    def weatherWetSpecularBoost(self) -> float: ...

    @weatherWetSpecularBoost.setter
    def weatherWetSpecularBoost(self, arg: float, /) -> None: ...

    @property
    def weatherSnowMaterialEnabled(self) -> bool: ...

    @weatherSnowMaterialEnabled.setter
    def weatherSnowMaterialEnabled(self, arg: bool, /) -> None: ...

    @property
    def weatherSnowCoverage(self) -> float: ...

    @weatherSnowCoverage.setter
    def weatherSnowCoverage(self, arg: float, /) -> None: ...

    @property
    def weatherSnowAccumulationStrength(self) -> float: ...

    @weatherSnowAccumulationStrength.setter
    def weatherSnowAccumulationStrength(self, arg: float, /) -> None: ...

    @property
    def weatherSnowAlbedoBlend(self) -> float: ...

    @weatherSnowAlbedoBlend.setter
    def weatherSnowAlbedoBlend(self, arg: float, /) -> None: ...

    @property
    def weatherSnowRoughness(self) -> float: ...

    @weatherSnowRoughness.setter
    def weatherSnowRoughness(self, arg: float, /) -> None: ...

    @property
    def weatherSnowMetallicScale(self) -> float: ...

    @weatherSnowMetallicScale.setter
    def weatherSnowMetallicScale(self, arg: float, /) -> None: ...

    @property
    def weatherSnowNormalSoftening(self) -> float: ...

    @weatherSnowNormalSoftening.setter
    def weatherSnowNormalSoftening(self, arg: float, /) -> None: ...

    @property
    def foliageWindEnabled(self) -> bool: ...

    @foliageWindEnabled.setter
    def foliageWindEnabled(self, arg: bool, /) -> None: ...

    @property
    def foliageWindDirection(self) -> numpy.ndarray: ...

    @foliageWindDirection.setter
    def foliageWindDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def foliageWindSpeed(self) -> float: ...

    @foliageWindSpeed.setter
    def foliageWindSpeed(self, arg: float, /) -> None: ...

    @property
    def foliageWindTimeSeconds(self) -> float: ...

    @foliageWindTimeSeconds.setter
    def foliageWindTimeSeconds(self, arg: float, /) -> None: ...

    @property
    def foliageWindGustStrength(self) -> float: ...

    @foliageWindGustStrength.setter
    def foliageWindGustStrength(self, arg: float, /) -> None: ...

    @property
    def foliageWindGustScale(self) -> float: ...

    @foliageWindGustScale.setter
    def foliageWindGustScale(self, arg: float, /) -> None: ...

    @property
    def foliageWindBranchBend(self) -> float: ...

    @foliageWindBranchBend.setter
    def foliageWindBranchBend(self, arg: float, /) -> None: ...

    @property
    def foliageWindLeafFlutter(self) -> float: ...

    @foliageWindLeafFlutter.setter
    def foliageWindLeafFlutter(self, arg: float, /) -> None: ...

    @property
    def addViewerFillLights(self) -> bool: ...

    @addViewerFillLights.setter
    def addViewerFillLights(self, arg: bool, /) -> None: ...

    def shadows(self) -> RenderQualitySettings.ShadowsSettings:
        """
        /// @brief Read a snapshot of the shadows sub-config from the flat fields.
          /// @return Copy of the main shadow and cascade fields (additional-light shadow fields excluded).
        C++: ShadowsSettings () const
        """

    def setShadows(self, s: RenderQualitySettings.ShadowsSettings) -> None:
        """
        /// @brief Write the shadows sub-config back into the flat fields.
          /// @param s Values to store.
        C++: void (const ShadowsSettings &)
        """

    def bloom(self) -> RenderQualitySettings.BloomSettings:
        """
        /// @brief Read a snapshot of the bloom sub-config from the flat fields.
          /// @return Copy of the bloom* fields.
        C++: BloomSettings () const
        """

    def setBloom(self, b: RenderQualitySettings.BloomSettings) -> None:
        """
        /// @brief Write the bloom sub-config back into the flat fields.
          /// @param b Values to store.
        C++: void (const BloomSettings &)
        """

    def depthOfField(self) -> RenderQualitySettings.DepthOfFieldSettings:
        """
        /// @brief Read a snapshot of the depth-of-field sub-config from the flat fields.
          /// @return Copy of the depthOfField* fields.
        C++: DepthOfFieldSettings () const
        """

    def setDepthOfField(self, d: RenderQualitySettings.DepthOfFieldSettings) -> None:
        """
        /// @brief Write the depth-of-field sub-config back into the flat fields.
          /// @param d Values to store.
        C++: void (const DepthOfFieldSettings &)
        """

    def fog(self) -> RenderQualitySettings.FogSettings:
        """
        /// @brief Read a snapshot of the fog sub-config from the flat fields.
          /// @return Copy of the distance/height/volumetric fog fields (wind and animation excluded).
        C++: FogSettings () const
        """

    def setFog(self, f: RenderQualitySettings.FogSettings) -> None:
        """
        /// @brief Write the fog sub-config back into the flat fields.
          /// @param f Values to store.
        C++: void (const FogSettings &)
        """

    def toneMapping(self) -> RenderQualitySettings.ToneMappingSettings:
        """
        /// @brief Read a snapshot of the tone-mapping and color sub-config from the flat fields.
          /// @return Copy of the tone-mapping, auto-exposure and color-grade fields (colorMode excluded).
        C++: ToneMappingSettings () const
        """

    def setToneMapping(self, t: RenderQualitySettings.ToneMappingSettings) -> None:
        """
        /// @brief Write the tone-mapping and color sub-config back into the flat fields.
          /// @param t Values to store.
        C++: void (const ToneMappingSettings &)
        """

    def temporalAa(self) -> RenderQualitySettings.TemporalAaSettings:
        """
        /// @brief Read a snapshot of the temporal-AA sub-config from the flat fields.
          /// @return Copy of the temporalAa* fields.
        C++: TemporalAaSettings () const
        """

    def setTemporalAa(self, a: RenderQualitySettings.TemporalAaSettings) -> None:
        """
        /// @brief Write the temporal-AA sub-config back into the flat fields.
          /// @param a Values to store.
        C++: void (const TemporalAaSettings &)
        """

    @staticmethod
    def preset(p: RenderQualityPreset) -> RenderQualitySettings:
        """
        /**
           * @brief Build a `RenderQualitySettings` for the given preset.
           *
           * Companion to the existing `RenderQualityPreset` enum so the type that
           * names the presets can also produce them. Delegates to
           * `RayraiWindow::defaultRenderQualitySettings()` (defined in
           * RenderQualityDefaults.cpp). The result is not clamped until it is passed to
           * `RayraiWindow::setRenderQualitySettings()`.
           *
           * @param p Preset to build.
           * @return Struct defaults with the preset's overrides applied.
           */
        C++: RenderQualitySettings (RenderQualityPreset)
        """

class SkyVisibilityGrid:
    def __init__(self) -> None: ...

    @property
    def dimensions(self) -> numpy.ndarray: ...

    @dimensions.setter
    def dimensions(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def minimum(self) -> numpy.ndarray: ...

    @minimum.setter
    def minimum(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def maximum(self) -> numpy.ndarray: ...

    @maximum.setter
    def maximum(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def values(self) -> list[float]: ...

    @values.setter
    def values(self, arg: Sequence[float], /) -> None: ...

    def validate(self) -> None:
        """C++: void () const"""

class TemporalAaMethod(enum.Enum):
    Legacy = 0

    Reprojected = 1

class ViewerColorMode(enum.Enum):
    FastLinear = 0

    AcesApprox = 1

    UnrealPreviewApprox = 2

    FilmicApprox = 3

    AgXApprox = 4

class ViewerUpscaleFilter(enum.Enum):
    Linear = 0

    Mitchell = 1

class VisualCategory(enum.Enum):
    NotDetectable = 0

    Detectable = 1

class Visuals:
    @overload
    def __init__(self, viewer: RayraiWindow, type: raisimpy.ShapeType, name: str, assets: RayraiGlobalAsset, meshFile: str = '') -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, name: str, meshes: Sequence[OpenGLMesh]) -> None: ...

    @overload
    def __init__(self, viewer: RayraiWindow, name: str, meshes: Sequence[OpenGLMesh], automaticMeshLodEnabled: bool) -> None: ...

    class FrameCacheMaterialState:
        def __init__(self) -> None: ...

        @property
        def cacheable(self) -> bool: ...

        @cacheable.setter
        def cacheable(self, arg: bool, /) -> None: ...

        @property
        def hasTextures(self) -> bool: ...

        @hasTextures.setter
        def hasTextures(self, arg: bool, /) -> None: ...

        @property
        def hasPbrMeshes(self) -> bool: ...

        @hasPbrMeshes.setter
        def hasPbrMeshes(self, arg: bool, /) -> None: ...

        @property
        def transparent(self) -> bool: ...

        @transparent.setter
        def transparent(self, arg: bool, /) -> None: ...

        @property
        def specialBlendMode(self) -> bool: ...

        @specialBlendMode.setter
        def specialBlendMode(self, arg: bool, /) -> None: ...

        @property
        def renderPriority(self) -> int: ...

        @renderPriority.setter
        def renderPriority(self, arg: int, /) -> None: ...

        @property
        def sortKeys(self) -> Visuals.MaterialSortKeys: ...

        @sortKeys.setter
        def sortKeys(self, arg: Visuals.MaterialSortKeys, /) -> None: ...

    class FrameCacheMaterialStateMemo:
        def __init__(self) -> None: ...

    class MaterialSortKeys:
        def __init__(self) -> None: ...

        @property
        def all(self) -> int: ...

        @all.setter
        def all(self, arg: int, /) -> None: ...

        @property
        def simpleOnly(self) -> int: ...

        @simpleOnly.setter
        def simpleOnly(self, arg: int, /) -> None: ...

        @property
        def pbrOnly(self) -> int: ...

        @pbrOnly.setter
        def pbrOnly(self, arg: int, /) -> None: ...

    class ShadowCastingMode(enum.Enum):
        Off = 0

        On = 1

        DoubleSided = 2

        ShadowsOnly = 3

    class VisibilityRangeFadeMode(enum.Enum):
        Disabled = 0

        Self = 1

        Dependencies = 2

    def setSphereSize(self, radius: float) -> None:
        """
        /**
           * @brief Set sphere radius (uniform scale of the unit-radius sphere).
           * @param radius Sphere radius in metres.
           */
        C++: void (double)
        """

    def setBoxSize(self, x: float, y: float, z: float) -> None:
        """
        /**
           * @brief Set box dimensions (scale of the unit cube).
           * @param x Full box length along local X, in metres.
           * @param y Full box length along local Y, in metres.
           * @param z Full box length along local Z, in metres.
           */
        C++: void (double, double, double)
        """

    def setCylinderSize(self, radius: float, height: float) -> None:
        """
        /**
           * @brief Set cylinder dimensions (scale of the unit cylinder, axis along local Z).
           * @param radius Cylinder radius in metres.
           * @param height Cylinder height in metres.
           */
        C++: void (double, double)
        """

    def setCapsuleSize(self, radius: float, height: float) -> None:
        """
        /**
           * @brief Set capsule dimensions.
           *
           * For a Capsule visual this regenerates a private capsule mesh (only when the
           * size changes) and resets the scale to 1; for other types it sets the scale
           * to (radius, radius, height).
           * @param radius Capsule radius in metres.
           * @param height Height of the cylindrical section (excluding the caps), in metres.
           */
        C++: void (double, double)
        """

    def setMeshScale(self, x: float, y: float, z: float) -> None:
        """
        /**
           * @brief Set mesh scale factors (applied in the mesh's local frame).
           * @param x Scale in X.
           * @param y Scale in Y.
           * @param z Scale in Z.
           */
        C++: void (double, double, double)
        """

    def setDefaultPrimitivePbrEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Toggle High/Ultra default PBR shading for built-in primitive meshes.
           *
           * Edits the material of the shared primitive mesh, so every visual using the
           * same primitive is affected. No-op for mesh visuals.
           * @param enabled True to allow the PBR pass.
           */
        C++: void (bool)
        """

    def shapeType(self) -> raisimpy.ShapeType:
        """
        /// @return Shape type this visual was created with (or set via setShapeType()).
        C++: raisim::Shape::Type () const
        """

    def setShapeType(self, type: raisimpy.ShapeType) -> None:
        """
        /// Change the recorded shape type without touching the meshes; it affects
          /// shading type, bounds estimation and LOD eligibility.
          /// @param type New shape type.
        C++: void (raisim::Shape::Type)
        """

    def configureGroundMaterial(self, reflective: bool, roughness: float, metallic: float, proceduralChecker: bool = False, terrainTextures: RayraiWindow | None = None) -> None:
        """
        /// @brief Configure this visual's mesh material as a reflective/non-reflective
          /// ground, like RaisimObject::configureGroundMaterial(). Used for ground
          /// visuals created over the TCP path which don't go through RaisimObject.
          /// Differences: a reflective ground here gets IBL and environment strength 0
          /// (planar reflection only), and a Ground-type visual stops casting shadows.
          /// HeightMap visuals only get the terrain material reapplied, with the
          /// heightmap terrain textures of @p terrainTextures when it has any. Edits
          /// the mesh materials in place.
          /// @param reflective Use a reflective PBR ground with planar reflection.
          /// @param roughness Roughness, clamped to [0.04, 1] (reflective only).
          /// @param metallic Metallic factor, clamped to [0, 1] (reflective only).
          /// @param proceduralChecker Draw a procedural checker instead of the texture.
          /// @param terrainTextures Renderer whose heightmap terrain textures a HeightMap visual takes
          ///   (see RayraiWindow::applyHeightmapTerrainTextures()), or nullptr.
        C++: void (bool, float, float, bool, const RayraiWindow *)
        """

    @overload
    def setPosition(self, x: float, y: float, z: float) -> None:
        """
        /**
           * @brief Set position from components.
           * @param x X position in world coordinates (m).
           * @param y Y position in world coordinates (m).
           * @param z Z position in world coordinates (m).
           */
        C++: void (double, double, double)
        """

    @overload
    def setPosition(self, pos: numpy.ndarray) -> None:
        """
        /**
           * @brief Set position from a vector.
           * @param pos Position in world coordinates (m).
           */
        C++: void (const glm::vec3 &)
        """

    @overload
    def setOrientation(self, w: float, x: float, y: float, z: float) -> None:
        """
        /**
           * @brief Set orientation from wxyz quaternion.
           *
           * The quaternion is used as given (not normalized); pass a unit quaternion.
           * @param w Quaternion w.
           * @param x Quaternion x.
           * @param y Quaternion y.
           * @param z Quaternion z.
           */
        C++: void (double, double, double, double)
        """

    @overload
    def setOrientation(self, ori: numpy.ndarray) -> None:
        """
        /**
           * @brief Set orientation from wxyz quaternion vector.
           * @param ori Quaternion (w, x, y, z): ori.x is w and ori.w is z, the same layout
           *        getOrientation() returns. The identity is glm::vec4(1, 0, 0, 0).
           */
        C++: void (const glm::vec4 &)
        """

    @overload
    def setColor(self, r: float, g: float, b: float, a: float) -> None:
        """
        /**
           * @brief Set object color from components.
           * @param r Linear red channel in [0, 1].
           * @param g Linear green channel in [0, 1].
           * @param b Linear blue channel in [0, 1].
           * @param a Alpha channel in [0, 1]; values below 1 route the visual to the transparent pass.
           */
        C++: void (double, double, double, double)
        """

    @overload
    def setColor(self, color: numpy.ndarray) -> None:
        """
        /**
           * @brief Set object color from vector.
           * @param color Linear RGBA color in [0, 1].
           */
        C++: void (const glm::vec4 &)
        """

    def setUseMeshColor(self, use: bool) -> None:
        """
        /**
           * @brief Use mesh-provided colors instead of object color.
           *
           * Untextured meshes then use their own base color (or material base color);
           * the visual's alpha still multiplies the result.
           * @param use True to use mesh colors.
           */
        C++: void (bool)
        """

    def usesMeshColor(self) -> bool:
        """
        /**
           * @brief Check if mesh colors are used.
           * @return True if mesh colors are used.
           */
        C++: bool () const
        """

    def setFlatShading(self, enabled: bool) -> None:
        """
        /**
           * @brief Force flat shading for this visual (overrides the shape's shading type).
           * @param enabled True to use flat shading.
           */
        C++: void (bool)
        """

    def isFlatShading(self) -> bool:
        """
        /**
           * @brief Check if flat shading is forced.
           * @return True if flat shading is used.
           */
        C++: bool () const
        """

    def setTwoSided(self, enabled: bool) -> None:
        """
        /**
           * @brief Disable face culling while drawing this visual.
           * @param enabled True to draw both sides of mesh triangles.
           */
        C++: void (bool)
        """

    def isTwoSided(self) -> bool:
        """
        /**
           * @brief Check if face culling is disabled for this visual.
           * @return True if both sides are drawn.
           */
        C++: bool () const
        """

    @overload
    def setPbrEnvironment(self, environmentMap: int, brdfLut: int = 0, strength: float = 1.0) -> None:
        """
        /**
           * @brief Apply an image-based lighting environment to all PBR submeshes.
           *
           * Same as the split overload with no irradiance or prefiltered map.
           * @param environmentMap GL cubemap id (0 clears).
           * @param brdfLut Split-sum BRDF LUT texture id, or 0.
           * @param strength Environment strength (Material::environmentMapStrength).
           */
        C++: void (unsigned int, unsigned int, float)
        """

    @overload
    def setPbrEnvironment(self, environmentMap: int, irradianceMap: int, prefilteredEnvironmentMap: int, brdfLut: int, strength: float = 1.0, boxProjection: bool = False, probePosition: numpy.ndarray = ..., boxMin: numpy.ndarray = ..., boxMax: numpy.ndarray = ...) -> None:
        """
        /**
           * @brief Apply split image-based lighting maps to all PBR submeshes.
           *
           * This mirrors the real-time IBL layout used by engines such as Filament: a
           * display/environment cubemap, diffuse irradiance, prefiltered specular
           * radiance, and a split-sum BRDF LUT. Supplying the full set gives whole-scene
           * imported visuals the same high-fidelity path used by hand-built PBR meshes.
           *
           * The ids are written into the materials of this visual's meshes (and LOD
           * meshes), material override, remaps and overlay, skipping materials that
           * look like plain simple-shaded ones; a non-zero cubemap also switches those
           * materials to Type::PBR. Mesh materials are edited in place, so other
           * visuals sharing the same cached meshes see the change. Texture ids are not owned.
           * @param environmentMap Radiance/display cubemap id.
           * @param irradianceMap Diffuse irradiance cubemap id, or 0.
           * @param prefilteredEnvironmentMap Prefiltered specular cubemap id, or 0.
           * @param brdfLut Split-sum BRDF LUT texture id, or 0.
           * @param strength Environment strength (capped further for heightmap terrain).
           * @param boxProjection Enable box-projected (parallax-corrected) reflections.
           * @param probePosition Probe capture position in world coordinates.
           * @param boxMin World-space minimum corner of the projection box.
           * @param boxMax World-space maximum corner of the projection box.
           */
        C++: void (unsigned int, unsigned int, unsigned int, unsigned int, float, bool, const glm::vec3 &, const glm::vec3 &, const glm::vec3 &)
        """

    @overload
    def setPbrEnvironment(self, env: PbrEnvironment) -> None:
        """
        /**
           * @brief Apply a PbrEnvironment bundle to all PBR submeshes.
           *
           * Convenience overload that unpacks the four texture handles and the
           * strength scalar from a single struct, eliminating the need for callers
           * to juggle four raw GL ids in parallel.
           * @param env Environment bundle (also carries the box-projection settings).
           */
        C++: void (const PbrEnvironment &)
        """

    def retainSceneHdrCubemaps(self, textures: Sequence[TextureHandle]) -> None:
        """
        /// Keep scene HDR cubemaps alive while this visual uses their texture ids.
        C++: void (std::array<std::shared_ptr<const unsigned int>, 3>)
        """

    def setCategory(self, category: VisualCategory) -> None:
        """
        /**
           * @brief Set the render category used by capture/detection filters.
           *
           * The category controls whether this user-created visual is included when a
           * rayrai pass excludes non-detectable visualization objects, such as external
           * camera scene-color renders, capture paths that reuse external-camera
           * rendering, or picking/shadow candidate policies that request detectable
           * custom visuals. The normal interactive viewer includes both categories by
           * default.
           *
           * This is render metadata only; it does not create collision, physics
           * behavior, or a semantic label. New `Visuals` default to
           * `VisualCategory::NotDetectable`.
           *
           * @param category New render category.
           * @see VisualCategory
           */
        C++: void (VisualCategory)
        """

    def setDetectable(self, detectable: bool) -> None:
        """
        /**
           * @brief Convenience helper for toggling the visual render category.
           * @param detectable True to include this visual in detectable-only
           *        capture/detection render passes; false to keep it viewer-only for
           *        those passes.
           */
        C++: void (bool)
        """

    def setSegmentationIds(self, objectIndex: int, bodyIndex: int, label: int) -> None:
        """
        /**
           * @brief Set the ids a segmentation render writes for this visual (RayraiWindow::renderSegmentation()).
           *
           * Visuals are not RaiSim objects, so they default to raisim::SegmentationCamera::kNotWorldObject (-2)
           * as object and body index and to label 0. The TCP viewer gives the visuals it builds for server
           * objects their server world index, body index and label.
           * @param objectIndex Object image value.
           * @param bodyIndex Body image value.
           * @param label Label image value.
           */
        C++: void (int32_t, int32_t, int32_t)
        """

    def setSegmentationLabel(self, label: int) -> None:
        """
        /**
           * @brief Set the label image value of this visual (see setSegmentationIds()).
           * @param label Label image value.
           */
        C++: void (int32_t)
        """

    def getSegmentationIds(self) -> numpy.ndarray:
        """
        /// @return (object index, body index, label, 0) written by segmentation renders.
        C++: glm::ivec4 () const
        """

    def getPosition(self) -> numpy.ndarray:
        """
        /**
           * @brief Get current position.
           * @return Position in world coordinates.
           */
        C++: glm::vec3 () const
        """

    def getOrientation(self) -> numpy.ndarray:
        """
        /**
           * @brief Get current orientation as a wxyz quaternion.
           * @return Quaternion with x holding w and w holding z.
           */
        C++: glm::vec4 () const
        """

    def getScale(self) -> numpy.ndarray:
        """
        /**
           * @brief Get current shape or mesh scale.
           * @return Per-axis scale (for primitives this encodes the size set by set*Size()).
           */
        C++: glm::vec3 () const
        """

    def getColor(self) -> numpy.ndarray:
        """
        /**
           * @brief Get current color.
           * @return Linear RGBA color in [0, 1].
           */
        C++: glm::vec4 () const
        """

    def getCategory(self) -> VisualCategory:
        """
        /**
           * @brief Get the render category used by capture/detection filters.
           * @return Current visual category.
           */
        C++: VisualCategory () const
        """

    def isDetectable(self) -> bool:
        """
        /**
           * @brief Check whether detectable-only render passes include this visual.
           * @return True when the category is `VisualCategory::Detectable`.
           */
        C++: bool () const
        """

    def setCastsShadows(self, castsShadows: bool) -> None:
        """
        /**
           * @brief Enable or disable rendering this visual into shadow maps.
           *
           * Large receiver-only helper surfaces can keep color rendering and shadow
           * receiving while staying out of the shadow caster pass.
           * @param castsShadows True for ShadowCastingMode::On, false for Off.
           */
        C++: void (bool)
        """

    def castsShadows(self) -> bool:
        """
        /**
           * @return True unless the shadow casting mode is Off.
           */
        C++: bool () const
        """

    def setShadowCastingMode(self, mode: Visuals.ShadowCastingMode) -> None:
        """
        /// @param mode New shadow casting mode.
        C++: void (ShadowCastingMode)
        """

    def shadowCastingMode(self) -> Visuals.ShadowCastingMode:
        """
        /// @return Current shadow casting mode.
        C++: ShadowCastingMode () const
        """

    def castsDoubleSidedShadows(self) -> bool:
        """
        /// @return True for ShadowCastingMode::DoubleSided.
        C++: bool () const
        """

    def setColorPassVisible(self, visible: bool) -> None:
        """
        /// Show or hide the visual in color passes without affecting shadow casting.
          /// @param visible False hides it from color passes.
        C++: void (bool)
        """

    def colorPassVisible(self) -> bool:
        """
        /// @return The flag set by setColorPassVisible().
        C++: bool () const
        """

    def visibleInColorPass(self) -> bool:
        """
        /// @return True when color-pass visible and not ShadowCastingMode::ShadowsOnly.
        C++: bool () const
        """

    def setVisibilityRange(self, begin: float, end: float, beginMargin: float = 0.0, endMargin: float = 0.0, fadeMode: Visuals.VisibilityRangeFadeMode = Visuals.VisibilityRangeFadeMode.Disabled) -> None:
        """
        /**
           * @brief Set renderer-side camera visibility range.
           *
           * A zero begin/end disables that side of the range. Self fade uses the
           * margins to alpha-fade this visual near the begin/end thresholds.
           * Distances are measured from the camera to the visual's bounds center, in
           * metres; negative inputs are clamped to 0. Visuals with self fade are drawn
           * in the transparent pass.
           * @param begin Hidden when closer than this.
           * @param end Hidden when farther than this.
           * @param beginMargin Half-width of the fade band around @p begin (Self mode).
           * @param endMargin Half-width of the fade band around @p end (Self mode).
           * @param fadeMode How the thresholds are applied.
           */
        C++: void (float, float, float, float, VisibilityRangeFadeMode)
        """

    def visibilityRangeBegin(self) -> float:
        """
        /// @return Near visibility threshold in metres (0 = none).
        C++: float () const
        """

    def visibilityRangeEnd(self) -> float:
        """
        /// @return Far visibility threshold in metres (0 = none).
        C++: float () const
        """

    def visibilityRangeBeginMargin(self) -> float:
        """
        /// @return Fade half-width around the near threshold.
        C++: float () const
        """

    def visibilityRangeEndMargin(self) -> float:
        """
        /// @return Fade half-width around the far threshold.
        C++: float () const
        """

    def visibilityRangeFadeMode(self) -> Visuals.VisibilityRangeFadeMode:
        """
        /// @return Current fade mode.
        C++: VisibilityRangeFadeMode () const
        """

    def hasVisibilityRange(self) -> bool:
        """
        /// @return True when either threshold is non-zero.
        C++: bool () const
        """

    def visibilityRangeUsesSelfFade(self) -> bool:
        """
        /// @return True when Self fade is active with a non-zero margin on an enabled side.
        C++: bool () const
        """

    def visibilityRangeAlpha(self, viewPos: numpy.ndarray) -> float:
        """
        /// @param viewPos Camera position in world coordinates.
          /// @return Visibility-range alpha in [0, 1] at @p viewPos (1 without a range).
        C++: float (const glm::vec3 &) const
        """

    def isVisibleAt(self, viewPos: numpy.ndarray) -> bool:
        """
        /// @param viewPos Camera position in world coordinates.
          /// @return True when visibilityRangeAlpha() exceeds 0.001.
        C++: bool (const glm::vec3 &) const
        """

    @overload
    def setMaterialOverride(self, material: Material) -> None:
        """
        /**
           * @brief Override the visual's draw material with a shared instance.
           *
           * **Ownership.** The same `shared_ptr<Material>` may be held by any number
           * of other visuals — `setMaterialOverride` does not deep-copy. Mutating
           * the pointed-to material from any holder is observed by every visual
           * that shares it. If you want an in-place edit that affects only this
           * visual, call the `const Material&` overload (which copies into a new
           * shared instance) or grab a fresh copy via `*materialOverride()` and
           * push it back through `setMaterialOverride(Material{...})`.
           *
           * The override replaces the material of every submesh (and takes precedence
           * over material remaps); the meshes' own #OpenGLMesh::textures are then ignored.
           * @param material Material to draw with; null clears the override.
           */
        C++: void (std::shared_ptr<Material>)
        """

    @overload
    def setMaterialOverride(self, material: Material) -> None:
        """
        /**
           * @brief Override the visual's draw material with a private copy.
           *
           * Copies @p material into a freshly allocated shared instance so later
           * mutations affect only this visual. Prefer this overload when the
           * material is going to be tweaked per-visual.
           * @param material Material to copy.
           */
        C++: void (const Material &)
        """

    def clearMaterialOverride(self) -> None:
        """
        /// Remove the material override; submeshes use their own (or remapped) materials again.
        C++: void ()
        """

    def hasMaterialOverride(self) -> bool:
        """
        /// @return True when a material override is set.
        C++: bool () const
        """

    def materialOverride(self) -> Material | None:
        """
        /// @return The override material, or nullptr. Owned by this visual (possibly shared).
        C++: const Material *() const
        """

    def mutableMaterialOverride(self) -> Material | None:
        """
        /**
           * @brief Mutable accessor for the override material.
           *
           * Honest companion to `setMaterialOverride(std::shared_ptr<Material>)`:
           * if the material is shared with other visuals, edits made through this
           * pointer are visible on all of them. Call notifyMeshMaterialsChanged()
           * after editing so renderer caches are refreshed.
           * @return The override material, or nullptr.
           */
        C++: Material *()
        """

    @overload
    def setMaterialRemap(self, sourceName: str, material: Material) -> None:
        """
        /**
           * @brief Replace the material of matching submeshes.
           *
           * A submesh matches when its Material::name or OpenGLMesh::debugSourceMeshName
           * equals @p sourceName; the key "*" matches every submesh not matched by
           * name. Ignored while a material override is set. The material is shared, not copied.
           * @param sourceName Material or source-mesh name; empty is ignored.
           * @param material Replacement material; null is ignored.
           */
        C++: void (const std::string &, std::shared_ptr<Material>)
        """

    @overload
    def setMaterialRemap(self, sourceName: str, material: Material) -> None:
        """
        /// Copying overload of setMaterialRemap().
          /// @param sourceName Material or source-mesh name; empty is ignored.
          /// @param material Material to copy.
        C++: void (const std::string &, const Material &)
        """

    def clearMaterialRemaps(self) -> None:
        """
        /// Remove all material remaps.
        C++: void ()
        """

    def materialRemapCount(self) -> int:
        """
        /// @return Number of registered remap keys.
        C++: size_t () const
        """

    @overload
    def setMaterialOverlay(self, material: Material) -> None:
        """
        /**
           * @brief Draw every submesh a second time with this material on top of its normal material.
           * @param material Overlay material (shared, not copied); null clears it.
           */
        C++: void (std::shared_ptr<Material>)
        """

    @overload
    def setMaterialOverlay(self, material: Material) -> None:
        """
        /// Copying overload of setMaterialOverlay().
          /// @param material Material to copy.
        C++: void (const Material &)
        """

    def clearMaterialOverlay(self) -> None:
        """
        /// Remove the overlay material.
        C++: void ()
        """

    def hasMaterialOverlay(self) -> bool:
        """
        /// @return True when an overlay material is set.
        C++: bool () const
        """

    def materialOverlay(self) -> Material | None:
        """
        /// @return The overlay material, or nullptr.
        C++: const Material *() const
        """

    def setTransparency(self, transparency: float) -> None:
        """
        /// Fade the whole visual: drawn alpha is multiplied by (1 - transparency).
          /// Values above 0 route the visual to the transparent pass; 1 skips drawing.
          /// @param transparency Value clamped to [0, 1].
        C++: void (float)
        """

    def transparency(self) -> float:
        """
        /// @return Transparency in [0, 1].
        C++: float () const
        """

    def setCustomBounds(self, localCenter: numpy.ndarray, localRadius: float) -> None:
        """
        /// Replace the culling bounds with a sphere given in the visual's local
          /// (pre-scale) frame; it is scaled by the largest scale component.
          /// @param localCenter Sphere center in local coordinates.
          /// @param localRadius Sphere radius; <= 0 disables custom bounds.
        C++: void (const glm::vec3 &, float)
        """

    def clearCustomBounds(self) -> None:
        """
        /// Return to automatic bounds.
        C++: void ()
        """

    def hasCustomBounds(self) -> bool:
        """
        /// @return True when custom bounds are active.
        C++: bool () const
        """

    def customBoundsLocalCenter(self) -> numpy.ndarray:
        """
        /// @return Custom-bounds center in local coordinates.
        C++: glm::vec3 () const
        """

    def customBoundsLocalRadius(self) -> float:
        """
        /// @return Custom-bounds radius in local units.
        C++: float () const
        """

    def setTransparentSortOffset(self, offset: float) -> None:
        """
        /// Bias the transparent sort: the sort distance is reduced by @p offset
          /// (clamped at 0), so a positive offset draws the visual later (as if closer).
          /// @param offset Distance offset in metres.
        C++: void (float)
        """

    def transparentSortOffset(self) -> float:
        """
        /// @return Transparent sort offset in metres.
        C++: float () const
        """

    def setTransparentSortUsesBoundsCenter(self, enabled: bool) -> None:
        """
        /// @param enabled True (default) to sort by the bounds center, false to sort by getPosition().
        C++: void (bool)
        """

    def transparentSortUsesBoundsCenter(self) -> bool:
        """
        /// @return Whether the transparent sort uses the bounds center.
        C++: bool () const
        """

    def transparentSortCenter(self) -> numpy.ndarray:
        """
        /// @return World point used for transparent sorting.
        C++: glm::vec3 () const
        """

    def transparentSortDistanceSquared(self, viewPos: numpy.ndarray) -> float:
        """
        /// @param viewPos Camera position in world coordinates.
          /// @return Squared sort distance after applying the sort offset.
        C++: float (const glm::vec3 &) const
        """

    def hasOpaqueDraws(self) -> bool:
        """
        /// @return True when some submesh is drawn in the opaque pass (false whenever
          ///         the visual alpha, transparency or self fade makes it transparent).
        C++: bool () const
        """

    def hasTransparentDraws(self) -> bool:
        """
        /// @return True when some submesh is drawn in the transparent pass.
        C++: bool () const
        """

    def isTransparent(self) -> bool:
        """
        /**
           * @brief Check if visual is translucent based on alpha.
           * @return True if the color alpha is below 1, transparency is non-zero,
           *         self fade is active, or any draw material needs the transparent pass.
           */
        C++: bool () const
        """

    def hasSpecialBlendMode(self) -> bool:
        """
        /**
           * @return True when at least one draw material uses a blend mode other than Material::BlendMode::Mix.
           */
        C++: bool () const
        """

    def renderPriority(self) -> int:
        """
        /**
           * @return Highest material render priority across the visual's draw materials (0 if none).
           */
        C++: int () const
        """

    def isValid(self) -> bool:
        """
        /**
           * @brief Check if the visual has valid mesh data.
           * @return True if a mesh list is assigned (it may still be empty while loading).
           */
        C++: bool () const
        """

    def renderRevision(self) -> int:
        """
        /// Internal: @return Counter bumped on every render-affecting change; used to invalidate frame caches.
        C++: uint64_t () const
        """

    def getName(self) -> str:
        """
        /**
           * @return The visual name used by the renderer.
           */
        C++: const std::string &() const
        """

    def approximateRadius(self) -> float:
        """
        /**
           * @return Radius from approximateBounds(), or 0 for an invalid visual.
           */
        C++: float () const
        """

    def approximateBounds(self) -> tuple:
        """
        /**
           * @brief Conservative world-space bounds for coarse culling.
           *
           * Uses custom bounds when set, mesh vertex bounds for Mesh/Ground/HeightMap
           * visuals, and otherwise a scale-based estimate around getPosition().
           * @param[out] center Output sphere center.
           * @param[out] radius Output sphere radius.
           * @return False for an invalid visual (outputs hold the scale-based estimate).
           */
        C++: bool (glm::vec3 &, float &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def shadowGeometryWithinBounds(self) -> bool:
        """
        /// @return True when shadow depth draws keep every vertex inside approximateBounds():
          ///         no mesh, override, remap or overlay material uses billboards,
          ///         fixed-size scaling or height displacement.
        C++: bool () const
        """

    def shadowDepthMayDiscard(self, transparentShadowDithering: bool) -> bool:
        """
        /// @return False only when no mesh (at any LOD), override, remap or overlay material can
          ///         make a shadow depth draw discard: none uses alpha mask, nor alpha blend while
          ///         @p transparentShadowDithering is set.
        C++: bool (bool) const
        """

    def shadowMaterialSignature(self) -> int:
        """
        /// Internal: hash of every material input of a shadow depth draw (mesh materials, textures and
          /// base colors, override, overlay and remap targets). OpenGLMesh::material and
          /// mutableMaterialOverride() are public and change no render revision, so cached shadow
          /// layers compare this too.
        C++: uint64_t () const
        """

    def hasTextures(self) -> bool:
        """
        /**
           * @brief Check if any mesh has textures bound.
           * @return True if a mesh (or the override/overlay/remapped material) references a texture.
           */
        C++: bool () const
        """

    def setAutomaticMeshLodEnabled(self, enabled: bool) -> None:
        """
        /**
           * @brief Enable automatic mesh LOD generation/selection for mesh visuals.
           *
           * This affects visuals created from addVisualMesh()/addVisualCustomMesh(). Built-in primitive
           * shapes keep their fixed authored meshes.
           * @param enabled True to generate (or fetch cached) LODs; false drops them.
           */
        C++: void (bool)
        """

    def automaticMeshLodEnabled(self) -> bool:
        """
        /**
           * @return True when automatic mesh LOD is enabled for this visual.
          */
        C++: bool () const
        """

    def setAutomaticMeshLodBias(self, bias: float) -> None:
        """
        /// Scale the projected size used for LOD selection; larger values keep finer LODs longer.
          /// @param bias Multiplier, clamped to [0.01, 16] (default 1).
        C++: void (float)
        """

    def automaticMeshLodBias(self) -> float:
        """
        /// @return Current LOD bias.
        C++: float () const
        """

    def automaticMeshLodLevelCount(self) -> int:
        """
        /**
           * @return Number of available mesh LOD levels including the source mesh at level 0 (0 without meshes).
           */
        C++: size_t () const
        """

    def automaticMeshLodTriangleCount(self, level: int) -> int:
        """
        /**
           * @param level LOD level; 0 is the source mesh.
           * @return Triangle count for one mesh LOD level, or 0 for an invalid level.
           */
        C++: size_t (size_t) const
        """

    def automaticMeshLodMaxUvClusterSpan(self, level: int) -> float:
        """
        /**
           * @param level Generated LOD level (>= 1).
           * @return Largest UV span collapsed into one generated LOD vertex (diagnostics); 0 for level 0.
           */
        C++: float (size_t) const
        """

    def selectAutomaticMeshLodLevel(self, viewPos: numpy.ndarray, verticalFovDeg: float, viewportHeight: int) -> int:
        """
        /**
           * @brief Select the mesh LOD level that would be used for a camera.
           * @param viewPos Camera position in world coordinates.
           * @param verticalFovDeg Camera vertical field of view in degrees.
           * @param viewportHeight Viewport height in pixels; <= 0 selects level 0.
           * @return Selected level (0 = source mesh).
           */
        C++: size_t (const glm::vec3 &, float, int) const
        """

    @overload
    def draw(self, shader: Shader, mode: OpenGLMesh.RenderMode = OpenGLMesh.RenderMode.All) -> None:
        """
        /**
           * @brief Draw the visual using a shader.
           *
           * Always uses the source mesh (no LOD) and ignores the visibility range.
           * Sets the `shadingType`, `model` and `objectColor` uniforms and draws each
           * submesh (plus the overlay material, if any) with OpenGLMesh::draw(). The
           * shader must already be in use; requires a current GL context.
           * @param shader Shader to use for drawing.
           * @param mode Material filter (see OpenGLMesh::RenderMode).
           */
        C++: void (Shader &, OpenGLMesh::RenderMode) const
        """

    @overload
    def draw(self, shader: Shader, viewPos: numpy.ndarray, verticalFovDeg: float, viewportHeight: int, mode: OpenGLMesh.RenderMode = OpenGLMesh.RenderMode.All) -> None:
        """
        /**
           * @brief Draw the visual using camera-aware automatic mesh LOD.
           *
           * Like draw(Shader&, OpenGLMesh::RenderMode) but selects a LOD level and
           * applies the visibility-range alpha for @p viewPos.
           * @param shader Shader that is already in use.
           * @param viewPos Camera position in world coordinates.
           * @param verticalFovDeg Camera vertical field of view in degrees.
           * @param viewportHeight Viewport height in pixels.
           * @param mode Material filter.
           */
        C++: void (Shader &, const glm::vec3 &, float, int, OpenGLMesh::RenderMode) const
        """

    @overload
    def drawWithBoundShader(self, shader: Shader, mode: OpenGLMesh.RenderMode = OpenGLMesh.RenderMode.All) -> None:
        """
        /**
           * @brief Same as draw(); assumes the caller has already bound @p shader.
           *
           * Currently equivalent to draw() because OpenGLMesh::draw also assumes the
           * shader is bound. Kept as an API hook for callers that opt in via
           * drawWithBoundShader.
           * @param shader Shader that is already in use.
           * @param mode Material filter.
           */
        C++: void (Shader &, OpenGLMesh::RenderMode) const
        """

    @overload
    def drawWithBoundShader(self, shader: Shader, viewPos: numpy.ndarray, verticalFovDeg: float, viewportHeight: int, mode: OpenGLMesh.RenderMode = OpenGLMesh.RenderMode.All) -> None:
        """
        /// Equivalent to the camera-aware draw(); kept for API parity.
          /// @param shader Shader that is already in use.
          /// @param viewPos Camera position in world coordinates.
          /// @param verticalFovDeg Camera vertical field of view in degrees.
          /// @param viewportHeight Viewport height in pixels.
          /// @param mode Material filter.
        C++: void (Shader &, const glm::vec3 &, float, int, OpenGLMesh::RenderMode) const
        """

    def drawSorted(self, shader: Shader, viewPos: numpy.ndarray, mode: OpenGLMesh.RenderMode = OpenGLMesh.RenderMode.All) -> None:
        """
        /**
           * @brief Draw submeshes back-to-front relative to a camera position.
           *
           * This is intended for transparent mesh visuals. It is more expensive than draw(), so rayrai
           * only uses it on the transparent visual path. Submeshes are ordered by
           * material render priority, then by distance (farthest first). Uses the
           * source mesh (no LOD) and applies the visibility-range alpha.
           * @param shader Shader that is already in use.
           * @param viewPos Camera position in world coordinates.
           * @param mode Material filter.
           */
        C++: void (Shader &, const glm::vec3 &, OpenGLMesh::RenderMode) const
        """

    def drawsInMode(self, mode: OpenGLMesh.RenderMode) -> bool:
        """
        /**
           * Whether drawSorted() in this mode would rasterize anything. Transparent
           * items are submitted once per shader, and a glass-only visual has nothing
           * for the simple and non-transmissive passes; asking first avoids their
           * program binds and per-object submission entirely.
           * @param mode Material filter.
           * @return True if some submesh (or overlay pass) would be drawn in @p mode.
           */
        C++: bool (OpenGLMesh::RenderMode) const
        """

    def hasPbrMeshes(self) -> bool:
        """
        /**
           * @name Material queries
           * Internal: renderer pass-selection queries over the visual's draw materials
           * (override, remaps and overlay included). Each mirrors the RaisimObject
           * query of the same name.
           */
          ///@{
          /// @return True when any draw material uses the PBR shader.
        C++: bool () const
        """

    def hasHighFidelityPbrMeshes(self) -> bool:
        """
        /// @return True when any PBR draw material requires the high-fidelity PBR path.
        C++: bool () const
        """

    def hasTransmissivePbrMeshes(self) -> bool:
        """
        /// @return True when any PBR draw material needs the transmission pass.
        C++: bool () const
        """

    def hasProximityFadePbrMeshes(self) -> bool:
        """
        /// @return True when any PBR draw material uses proximity fade.
        C++: bool () const
        """

    def usesDefaultDepthAndStencilState(self) -> bool:
        """
        /// @return True when every draw material uses default depth/stencil state.
        C++: bool () const
        """

    def hasFullPbrMeshes(self) -> bool:
        """
        /// @return True when any PBR draw material cannot use a compact PBR shader.
        C++: bool () const
        """

    def hasFoliageFastPbrMeshes(self) -> bool:
        """
        /// @return True when any draw material can use the foliage fast-PBR path.
        C++: bool () const
        """

    def hasWeatherEligiblePbrMeshes(self) -> bool:
        """
        /// @return True when any draw material reacts to active weather.
        C++: bool () const
        """

    def hasSimpleMeshes(self) -> bool:
        """
        /// @return True when any draw material uses the simple shader.
        C++: bool () const
        """

    def hasPlanarReflectionMeshes(self) -> bool:
        """
        /// @return True when any PBR draw material uses planar reflection.
        C++: bool () const
        """

    def hasProceduralGroundCheckerMeshes(self) -> bool:
        """
        /// @return True when any draw material enables Material::proceduralGroundChecker.
        C++: bool () const
        """

    def materialSortKey(self, mode: OpenGLMesh.RenderMode) -> int:
        """
        ///@}
          /**
           * @brief Compact key for sorting opaque visual draws by material/texture state.
           * @param mode Render mode the draws will use.
           * @return Hash-like key combining the submeshes' OpenGLMesh::materialSortKey(); sort only, not unique.
           */
        C++: uint64_t (OpenGLMesh::RenderMode) const
        """

    def frameCacheMaterialState(self, memo: Visuals.FrameCacheMaterialStateMemo | None = None) -> Visuals.FrameCacheMaterialState:
        """
        /// Internal: compute FrameCacheMaterialState in one traversal of the submeshes.
          /// @param memo Optional memo to read/write (ignored for non-default PBR environments).
          /// @return The summarized state.
        C++: FrameCacheMaterialState (FrameCacheMaterialStateMemo *) const
        """

    def setMeshes(self, meshes: Sequence[OpenGLMesh]) -> None:
        """
        /**
           * @brief Replace mesh data for this visual.
           *
           * The list is shared, not copied. Automatic LODs are rebuilt, asset-level LOD
           * caching is disabled for this visual, and the stored PBR environment is
           * reapplied to the new meshes.
           * @param meshes New mesh list to render; null makes the visual invalid.
           */
        C++: void (const std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> &)
        """

    def updateSingleMeshVertices(self, vertices: Sequence[Vertex]) -> bool:
        """
        /**
           * @brief Update the first mesh's vertex buffer while preserving its index buffer.
           * @param vertices New vertex data for the existing mesh.
           * @return True when a single mesh was present and updated.
           *
           * Tangents are recomputed and generated LODs are dropped. The mesh is edited
           * in place (shared users see it) and a GL context must be current.
           */
        C++: bool (const std::vector<Vertex> &)
        """

    def notifyMeshMaterialsChanged(self) -> None:
        """
        /** Notify renderer caches after in-place material changes to existing meshes.
           *  Also reapplies the stored PBR environment. */
        C++: void ()
        """

    def applyTextureFilteringQuality(self, anisotropy: float, mipLodBias: float) -> int:
        """
        /**
           * @brief Apply sampler quality to all 2D material textures in this visual.
           *
           * Includes generated LOD meshes; modifies the shared GL texture objects.
           * @param anisotropy Requested anisotropic filtering level.
           * @param mipLodBias Mip LOD bias.
           * @return Number of texture ids updated (summed per mesh).
           */
        C++: size_t (float, float) const
        """

    def collectMaterialTextureIds(self) -> tuple:
        """
        /**
           * @brief Append non-environment material texture ids used by this visual.
           * @param[in,out] textureIds Ids are appended (duplicates are kept).
           */
        C++: void (std::vector<unsigned int> &) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

    def accumulateGeometryDiagnostics(self, diagnostics: OpenGLMesh.GeometryDiagnostics, tinyTriangleAreaThreshold: float) -> None:
        """
        /**
           * @brief Accumulate CPU-side mesh geometry diagnostics for this visual.
           * @param[in,out] diagnostics Totals to add to.
           * @param tinyTriangleAreaThreshold Triangles with smaller (non-zero) area count as tiny.
           */
        C++: void (OpenGLMesh::GeometryDiagnostics &, float) const
        """

    def collectMeshDiagnostics(self, tinyTriangleAreaThreshold: float) -> tuple:
        """
        /**
           * @brief Append per-mesh geometry/material diagnostics for this visual.
           * @param[in,out] diagnostics One entry is appended per submesh.
           * @param tinyTriangleAreaThreshold Triangles with smaller (non-zero) area count as tiny.
           */
        C++: void (std::vector<OpenGLMesh::MeshDiagnostics> &, float) const
        Returns a tuple containing the native result (when present) followed by output parameters.
        """

class WaterBody:
    def __init__(self) -> None: ...

    @property
    def shape(self) -> WaterBodyShape: ...

    @shape.setter
    def shape(self, arg: WaterBodyShape, /) -> None: ...

    @property
    def center(self) -> numpy.ndarray: ...

    @center.setter
    def center(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def halfExtents(self) -> numpy.ndarray: ...

    @halfExtents.setter
    def halfExtents(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def depth(self) -> float: ...

    @depth.setter
    def depth(self, arg: float, /) -> None: ...

    @property
    def yawRadians(self) -> float: ...

    @yawRadians.setter
    def yawRadians(self, arg: float, /) -> None: ...

    @property
    def edgeFadeDistance(self) -> float: ...

    @edgeFadeDistance.setter
    def edgeFadeDistance(self, arg: float, /) -> None: ...

    @property
    def waves(self) -> list[GerstnerWave]: ...

    @waves.setter
    def waves(self, arg: Sequence[GerstnerWave], /) -> None: ...

    @property
    def absorption(self) -> numpy.ndarray: ...

    @absorption.setter
    def absorption(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def scattering(self) -> numpy.ndarray: ...

    @scattering.setter
    def scattering(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def scatteringAnisotropy(self) -> float: ...

    @scatteringAnisotropy.setter
    def scatteringAnisotropy(self, arg: float, /) -> None: ...

    @property
    def indexOfRefraction(self) -> float: ...

    @indexOfRefraction.setter
    def indexOfRefraction(self, arg: float, /) -> None: ...

    @property
    def roughness(self) -> float: ...

    @roughness.setter
    def roughness(self, arg: float, /) -> None: ...

    @property
    def rippleStrength(self) -> float: ...

    @rippleStrength.setter
    def rippleStrength(self, arg: float, /) -> None: ...

    @property
    def rippleScale(self) -> float: ...

    @rippleScale.setter
    def rippleScale(self, arg: float, /) -> None: ...

    @property
    def rippleDirection(self) -> numpy.ndarray: ...

    @rippleDirection.setter
    def rippleDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def foamColor(self) -> numpy.ndarray: ...

    @foamColor.setter
    def foamColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def shorelineFoamDepth(self) -> float: ...

    @shorelineFoamDepth.setter
    def shorelineFoamDepth(self, arg: float, /) -> None: ...

    @property
    def crestFoamStrength(self) -> float: ...

    @crestFoamStrength.setter
    def crestFoamStrength(self, arg: float, /) -> None: ...

    @property
    def crestFoamThreshold(self) -> float: ...

    @crestFoamThreshold.setter
    def crestFoamThreshold(self, arg: float, /) -> None: ...

    @property
    def causticStrength(self) -> float: ...

    @causticStrength.setter
    def causticStrength(self, arg: float, /) -> None: ...

    @property
    def causticScale(self) -> float: ...

    @causticScale.setter
    def causticScale(self, arg: float, /) -> None: ...

    @property
    def shorelineFadeDepth(self) -> float: ...

    @shorelineFadeDepth.setter
    def shorelineFadeDepth(self, arg: float, /) -> None: ...

    @property
    def flow(self) -> WaterFlow: ...

    @flow.setter
    def flow(self, arg: WaterFlow, /) -> None: ...

class WaterBodyShape(enum.Enum):
    Ocean = 0

    Rectangle = 1

class WaterDiagnostics:
    def __init__(self) -> None: ...

    @property
    def bodyCount(self) -> int: ...

    @bodyCount.setter
    def bodyCount(self, arg: int, /) -> None: ...

    @property
    def bodiesDrawn(self) -> int: ...

    @bodiesDrawn.setter
    def bodiesDrawn(self, arg: int, /) -> None: ...

    @property
    def tilesDrawn(self) -> int: ...

    @tilesDrawn.setter
    def tilesDrawn(self, arg: int, /) -> None: ...

    @property
    def trianglesDrawn(self) -> int: ...

    @trianglesDrawn.setter
    def trianglesDrawn(self, arg: int, /) -> None: ...

    @property
    def sceneCaptured(self) -> bool: ...

    @sceneCaptured.setter
    def sceneCaptured(self, arg: bool, /) -> None: ...

    @property
    def screenSpaceReflections(self) -> bool: ...

    @screenSpaceReflections.setter
    def screenSpaceReflections(self, arg: bool, /) -> None: ...

    @property
    def cameraUnderwater(self) -> bool: ...

    @cameraUnderwater.setter
    def cameraUnderwater(self, arg: bool, /) -> None: ...

    @property
    def underwaterFogDrawn(self) -> bool: ...

    @underwaterFogDrawn.setter
    def underwaterFogDrawn(self, arg: bool, /) -> None: ...

    @property
    def skyCaptureFaceSize(self) -> int: ...

    @skyCaptureFaceSize.setter
    def skyCaptureFaceSize(self, arg: int, /) -> None: ...

    @property
    def skyCaptures(self) -> int: ...

    @skyCaptures.setter
    def skyCaptures(self, arg: int, /) -> None: ...

    @property
    def timeSeconds(self) -> float: ...

    @timeSeconds.setter
    def timeSeconds(self, arg: float, /) -> None: ...

    @property
    def flowBodies(self) -> int: ...

    @flowBodies.setter
    def flowBodies(self, arg: int, /) -> None: ...

    @property
    def flowSteps(self) -> int: ...

    @flowSteps.setter
    def flowSteps(self, arg: int, /) -> None: ...

    @property
    def flowTimeDropped(self) -> float: ...

    @flowTimeDropped.setter
    def flowTimeDropped(self, arg: float, /) -> None: ...

class WaterFlow:
    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def cellSize(self) -> float: ...

    @cellSize.setter
    def cellSize(self, arg: float, /) -> None: ...

    @property
    def manningRoughness(self) -> float: ...

    @manningRoughness.setter
    def manningRoughness(self, arg: float, /) -> None: ...

    @property
    def edges(self) -> WaterFlowEdges: ...

    @edges.setter
    def edges(self, arg: WaterFlowEdges, /) -> None: ...

    @property
    def bedCeiling(self) -> float: ...

    @bedCeiling.setter
    def bedCeiling(self, arg: float, /) -> None: ...

    @property
    def movingBodies(self) -> bool: ...

    @movingBodies.setter
    def movingBodies(self, arg: bool, /) -> None: ...

    @property
    def foamStrength(self) -> float: ...

    @foamStrength.setter
    def foamStrength(self, arg: float, /) -> None: ...

    @property
    def sources(self) -> list[WaterFlowSource]: ...

    @sources.setter
    def sources(self, arg: Sequence[WaterFlowSource], /) -> None: ...

class WaterFlowEdges(enum.Enum):
    Open = 0

    Closed = 1

class WaterFlowSample:
    def __init__(self) -> None: ...

    @property
    def inside(self) -> bool: ...

    @inside.setter
    def inside(self, arg: bool, /) -> None: ...

    @property
    def wet(self) -> bool: ...

    @wet.setter
    def wet(self, arg: bool, /) -> None: ...

    @property
    def surfaceHeight(self) -> float: ...

    @surfaceHeight.setter
    def surfaceHeight(self, arg: float, /) -> None: ...

    @property
    def bedHeight(self) -> float: ...

    @bedHeight.setter
    def bedHeight(self, arg: float, /) -> None: ...

    @property
    def depth(self) -> float: ...

    @depth.setter
    def depth(self, arg: float, /) -> None: ...

    @property
    def velocity(self) -> numpy.ndarray: ...

    @velocity.setter
    def velocity(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def timeSeconds(self) -> float: ...

    @timeSeconds.setter
    def timeSeconds(self, arg: float, /) -> None: ...

class WaterFlowSource:
    def __init__(self) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def flowRate(self) -> float: ...

    @flowRate.setter
    def flowRate(self, arg: float, /) -> None: ...

    @property
    def velocity(self) -> numpy.ndarray: ...

    @velocity.setter
    def velocity(self, arg: numpy.ndarray, /) -> None: ...

class WaterPreset(enum.Enum):
    Ocean = 0

    CalmLake = 1

    Pool = 2

    MurkyPond = 3

class WaterSurfaceSample:
    def __init__(self) -> None: ...

    @property
    def inside(self) -> bool: ...

    @inside.setter
    def inside(self, arg: bool, /) -> None: ...

    @property
    def height(self) -> float: ...

    @height.setter
    def height(self, arg: float, /) -> None: ...

    @property
    def normal(self) -> numpy.ndarray: ...

    @normal.setter
    def normal(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def compression(self) -> float: ...

    @compression.setter
    def compression(self, arg: float, /) -> None: ...

class WaterWaveSpectrum:
    def __init__(self) -> None: ...

    @property
    def waveCount(self) -> int: ...

    @waveCount.setter
    def waveCount(self, arg: int, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def windAngleRadians(self) -> float: ...

    @windAngleRadians.setter
    def windAngleRadians(self, arg: float, /) -> None: ...

    @property
    def directionSpreadRadians(self) -> float: ...

    @directionSpreadRadians.setter
    def directionSpreadRadians(self, arg: float, /) -> None: ...

    @property
    def minWavelength(self) -> float: ...

    @minWavelength.setter
    def minWavelength(self, arg: float, /) -> None: ...

    @property
    def maxWavelength(self) -> float: ...

    @maxWavelength.setter
    def maxWavelength(self, arg: float, /) -> None: ...

    @property
    def wavelengthFalloff(self) -> float: ...

    @wavelengthFalloff.setter
    def wavelengthFalloff(self, arg: float, /) -> None: ...

    @property
    def minAmplitude(self) -> float: ...

    @minAmplitude.setter
    def minAmplitude(self, arg: float, /) -> None: ...

    @property
    def maxAmplitude(self) -> float: ...

    @maxAmplitude.setter
    def maxAmplitude(self, arg: float, /) -> None: ...

    @property
    def amplitudeFalloff(self) -> float: ...

    @amplitudeFalloff.setter
    def amplitudeFalloff(self, arg: float, /) -> None: ...

    @property
    def longWaveSteepness(self) -> float: ...

    @longWaveSteepness.setter
    def longWaveSteepness(self, arg: float, /) -> None: ...

    @property
    def shortWaveSteepness(self) -> float: ...

    @shortWaveSteepness.setter
    def shortWaveSteepness(self, arg: float, /) -> None: ...

    @property
    def steepnessFalloff(self) -> float: ...

    @steepnessFalloff.setter
    def steepnessFalloff(self, arg: float, /) -> None: ...

    @property
    def randomness(self) -> float: ...

    @randomness.setter
    def randomness(self, arg: float, /) -> None: ...

    @property
    def maxSteepnessSum(self) -> float: ...

    @maxSteepnessSum.setter
    def maxSteepnessSum(self, arg: float, /) -> None: ...

class WeatherDiagnostics:
    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def viewerEffectsActive(self) -> bool: ...

    @viewerEffectsActive.setter
    def viewerEffectsActive(self, arg: bool, /) -> None: ...

    @property
    def affectsSensors(self) -> bool: ...

    @affectsSensors.setter
    def affectsSensors(self, arg: bool, /) -> None: ...

    @property
    def preset(self) -> WeatherPreset: ...

    @preset.setter
    def preset(self, arg: WeatherPreset, /) -> None: ...

    @property
    def quality(self) -> WeatherQuality: ...

    @quality.setter
    def quality(self, arg: WeatherQuality, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def timeOfDayHours(self) -> float: ...

    @timeOfDayHours.setter
    def timeOfDayHours(self, arg: float, /) -> None: ...

    @property
    def latitude(self) -> float: ...

    @latitude.setter
    def latitude(self, arg: float, /) -> None: ...

    @property
    def longitude(self) -> float: ...

    @longitude.setter
    def longitude(self, arg: float, /) -> None: ...

    @property
    def automaticUtcOffset(self) -> bool: ...

    @automaticUtcOffset.setter
    def automaticUtcOffset(self, arg: bool, /) -> None: ...

    @property
    def utcOffsetHours(self) -> float: ...

    @utcOffsetHours.setter
    def utcOffsetHours(self, arg: float, /) -> None: ...

    @property
    def resolvedUtcOffsetHours(self) -> float: ...

    @resolvedUtcOffsetHours.setter
    def resolvedUtcOffsetHours(self, arg: float, /) -> None: ...

    @property
    def year(self) -> int: ...

    @year.setter
    def year(self, arg: int, /) -> None: ...

    @property
    def month(self) -> int: ...

    @month.setter
    def month(self, arg: int, /) -> None: ...

    @property
    def day(self) -> int: ...

    @day.setter
    def day(self, arg: int, /) -> None: ...

    @property
    def solarAzimuthDegrees(self) -> float: ...

    @solarAzimuthDegrees.setter
    def solarAzimuthDegrees(self, arg: float, /) -> None: ...

    @property
    def solarElevationDegrees(self) -> float: ...

    @solarElevationDegrees.setter
    def solarElevationDegrees(self, arg: float, /) -> None: ...

    @property
    def sunDirection(self) -> numpy.ndarray: ...

    @sunDirection.setter
    def sunDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def sunColor(self) -> numpy.ndarray: ...

    @sunColor.setter
    def sunColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def sunIntensity(self) -> float: ...

    @sunIntensity.setter
    def sunIntensity(self, arg: float, /) -> None: ...

    @property
    def moonLightActive(self) -> bool: ...

    @moonLightActive.setter
    def moonLightActive(self, arg: bool, /) -> None: ...

    @property
    def moonDirection(self) -> numpy.ndarray: ...

    @moonDirection.setter
    def moonDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def moonIntensity(self) -> float: ...

    @moonIntensity.setter
    def moonIntensity(self, arg: float, /) -> None: ...

    @property
    def skyExposure(self) -> float: ...

    @skyExposure.setter
    def skyExposure(self, arg: float, /) -> None: ...

    @property
    def cloudCoverage(self) -> float: ...

    @cloudCoverage.setter
    def cloudCoverage(self, arg: float, /) -> None: ...

    @property
    def cloudDensity(self) -> float: ...

    @cloudDensity.setter
    def cloudDensity(self, arg: float, /) -> None: ...

    @property
    def cloudLayerActive(self) -> bool: ...

    @cloudLayerActive.setter
    def cloudLayerActive(self, arg: bool, /) -> None: ...

    @property
    def cloudShadowActive(self) -> bool: ...

    @cloudShadowActive.setter
    def cloudShadowActive(self, arg: bool, /) -> None: ...

    @property
    def cloudAltitudeMeters(self) -> float: ...

    @cloudAltitudeMeters.setter
    def cloudAltitudeMeters(self, arg: float, /) -> None: ...

    @property
    def cloudThicknessMeters(self) -> float: ...

    @cloudThicknessMeters.setter
    def cloudThicknessMeters(self, arg: float, /) -> None: ...

    @property
    def cloudShadowStrength(self) -> float: ...

    @cloudShadowStrength.setter
    def cloudShadowStrength(self, arg: float, /) -> None: ...

    @property
    def cloudScale(self) -> float: ...

    @cloudScale.setter
    def cloudScale(self, arg: float, /) -> None: ...

    @property
    def cloudOffset(self) -> numpy.ndarray: ...

    @cloudOffset.setter
    def cloudOffset(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def fogDensity(self) -> float: ...

    @fogDensity.setter
    def fogDensity(self, arg: float, /) -> None: ...

    @property
    def visibilityMeters(self) -> float: ...

    @visibilityMeters.setter
    def visibilityMeters(self, arg: float, /) -> None: ...

    @property
    def heightFogActive(self) -> bool: ...

    @heightFogActive.setter
    def heightFogActive(self, arg: bool, /) -> None: ...

    @property
    def volumetricFogActive(self) -> bool: ...

    @volumetricFogActive.setter
    def volumetricFogActive(self, arg: bool, /) -> None: ...

    @property
    def heightFogDensity(self) -> float: ...

    @heightFogDensity.setter
    def heightFogDensity(self, arg: float, /) -> None: ...

    @property
    def heightFogBaseHeight(self) -> float: ...

    @heightFogBaseHeight.setter
    def heightFogBaseHeight(self, arg: float, /) -> None: ...

    @property
    def heightFogFalloff(self) -> float: ...

    @heightFogFalloff.setter
    def heightFogFalloff(self, arg: float, /) -> None: ...

    @property
    def fogColor(self) -> numpy.ndarray: ...

    @fogColor.setter
    def fogColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def fogAnisotropy(self) -> float: ...

    @fogAnisotropy.setter
    def fogAnisotropy(self, arg: float, /) -> None: ...

    @property
    def visibilityTransmittance100m(self) -> float: ...

    @visibilityTransmittance100m.setter
    def visibilityTransmittance100m(self, arg: float, /) -> None: ...

    @property
    def visibilityTransmittance1km(self) -> float: ...

    @visibilityTransmittance1km.setter
    def visibilityTransmittance1km(self, arg: float, /) -> None: ...

    @property
    def localFogVolumeBudget(self) -> int: ...

    @localFogVolumeBudget.setter
    def localFogVolumeBudget(self, arg: int, /) -> None: ...

    @property
    def localFogVolumeCount(self) -> int: ...

    @localFogVolumeCount.setter
    def localFogVolumeCount(self, arg: int, /) -> None: ...

    @property
    def localFogPositiveDensity(self) -> float: ...

    @localFogPositiveDensity.setter
    def localFogPositiveDensity(self, arg: float, /) -> None: ...

    @property
    def localFogNegativeDensity(self) -> float: ...

    @localFogNegativeDensity.setter
    def localFogNegativeDensity(self, arg: float, /) -> None: ...

    @property
    def precipitationRate(self) -> float: ...

    @precipitationRate.setter
    def precipitationRate(self, arg: float, /) -> None: ...

    @property
    def effectivePrecipitationRate(self) -> float: ...

    @effectivePrecipitationRate.setter
    def effectivePrecipitationRate(self, arg: float, /) -> None: ...

    @property
    def rainOcclusionStrength(self) -> float: ...

    @rainOcclusionStrength.setter
    def rainOcclusionStrength(self, arg: float, /) -> None: ...

    @property
    def wetness(self) -> float: ...

    @wetness.setter
    def wetness(self, arg: float, /) -> None: ...

    @property
    def wetnessAccumulationActive(self) -> bool: ...

    @wetnessAccumulationActive.setter
    def wetnessAccumulationActive(self, arg: bool, /) -> None: ...

    @property
    def accumulatedWetness(self) -> float: ...

    @accumulatedWetness.setter
    def accumulatedWetness(self, arg: float, /) -> None: ...

    @property
    def wetnessAccumulationRate(self) -> float: ...

    @wetnessAccumulationRate.setter
    def wetnessAccumulationRate(self, arg: float, /) -> None: ...

    @property
    def wetnessDryingRate(self) -> float: ...

    @wetnessDryingRate.setter
    def wetnessDryingRate(self, arg: float, /) -> None: ...

    @property
    def wetMaterialActive(self) -> bool: ...

    @wetMaterialActive.setter
    def wetMaterialActive(self, arg: bool, /) -> None: ...

    @property
    def wetMaterialStrength(self) -> float: ...

    @wetMaterialStrength.setter
    def wetMaterialStrength(self, arg: float, /) -> None: ...

    @property
    def puddleStrength(self) -> float: ...

    @puddleStrength.setter
    def puddleStrength(self, arg: float, /) -> None: ...

    @property
    def rainRippleStrength(self) -> float: ...

    @rainRippleStrength.setter
    def rainRippleStrength(self, arg: float, /) -> None: ...

    @property
    def wetAlbedoDarkening(self) -> float: ...

    @wetAlbedoDarkening.setter
    def wetAlbedoDarkening(self, arg: float, /) -> None: ...

    @property
    def wetRoughnessScale(self) -> float: ...

    @wetRoughnessScale.setter
    def wetRoughnessScale(self, arg: float, /) -> None: ...

    @property
    def wetSpecularBoost(self) -> float: ...

    @wetSpecularBoost.setter
    def wetSpecularBoost(self, arg: float, /) -> None: ...

    @property
    def snowCoverage(self) -> float: ...

    @snowCoverage.setter
    def snowCoverage(self, arg: float, /) -> None: ...

    @property
    def snowMeltActive(self) -> bool: ...

    @snowMeltActive.setter
    def snowMeltActive(self, arg: bool, /) -> None: ...

    @property
    def snowMeltFactor(self) -> float: ...

    @snowMeltFactor.setter
    def snowMeltFactor(self, arg: float, /) -> None: ...

    @property
    def snowMeltWetnessBoost(self) -> float: ...

    @snowMeltWetnessBoost.setter
    def snowMeltWetnessBoost(self, arg: float, /) -> None: ...

    @property
    def snowMaterialActive(self) -> bool: ...

    @snowMaterialActive.setter
    def snowMaterialActive(self, arg: bool, /) -> None: ...

    @property
    def snowMaterialStrength(self) -> float: ...

    @snowMaterialStrength.setter
    def snowMaterialStrength(self, arg: float, /) -> None: ...

    @property
    def snowAccumulationStrength(self) -> float: ...

    @snowAccumulationStrength.setter
    def snowAccumulationStrength(self, arg: float, /) -> None: ...

    @property
    def snowAlbedoBlend(self) -> float: ...

    @snowAlbedoBlend.setter
    def snowAlbedoBlend(self, arg: float, /) -> None: ...

    @property
    def snowRoughness(self) -> float: ...

    @snowRoughness.setter
    def snowRoughness(self, arg: float, /) -> None: ...

    @property
    def snowMetallicScale(self) -> float: ...

    @snowMetallicScale.setter
    def snowMetallicScale(self, arg: float, /) -> None: ...

    @property
    def snowNormalSoftening(self) -> float: ...

    @snowNormalSoftening.setter
    def snowNormalSoftening(self, arg: float, /) -> None: ...

    @property
    def lightningRate(self) -> float: ...

    @lightningRate.setter
    def lightningRate(self, arg: float, /) -> None: ...

    @property
    def lightningActive(self) -> bool: ...

    @lightningActive.setter
    def lightningActive(self, arg: bool, /) -> None: ...

    @property
    def lightningEventIndex(self) -> int: ...

    @lightningEventIndex.setter
    def lightningEventIndex(self, arg: int, /) -> None: ...

    @property
    def lightningEventTimeSeconds(self) -> float: ...

    @lightningEventTimeSeconds.setter
    def lightningEventTimeSeconds(self, arg: float, /) -> None: ...

    @property
    def lightningNextEventSeconds(self) -> float: ...

    @lightningNextEventSeconds.setter
    def lightningNextEventSeconds(self, arg: float, /) -> None: ...

    @property
    def lightningEventPhaseSeconds(self) -> float: ...

    @lightningEventPhaseSeconds.setter
    def lightningEventPhaseSeconds(self, arg: float, /) -> None: ...

    @property
    def lightningEventDurationSeconds(self) -> float: ...

    @lightningEventDurationSeconds.setter
    def lightningEventDurationSeconds(self, arg: float, /) -> None: ...

    @property
    def lightningIntensity(self) -> float: ...

    @lightningIntensity.setter
    def lightningIntensity(self, arg: float, /) -> None: ...

    @property
    def lightningExposureBoost(self) -> float: ...

    @lightningExposureBoost.setter
    def lightningExposureBoost(self, arg: float, /) -> None: ...

    @property
    def lightningSkyFlash(self) -> float: ...

    @lightningSkyFlash.setter
    def lightningSkyFlash(self, arg: float, /) -> None: ...

    @property
    def lightningDirection(self) -> numpy.ndarray: ...

    @lightningDirection.setter
    def lightningDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lightningColor(self) -> numpy.ndarray: ...

    @lightningColor.setter
    def lightningColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lightningPointLightActive(self) -> bool: ...

    @lightningPointLightActive.setter
    def lightningPointLightActive(self, arg: bool, /) -> None: ...

    @property
    def lightningPointLightPosition(self) -> numpy.ndarray: ...

    @lightningPointLightPosition.setter
    def lightningPointLightPosition(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lightningPointLightColor(self) -> numpy.ndarray: ...

    @lightningPointLightColor.setter
    def lightningPointLightColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lightningPointLightRadius(self) -> float: ...

    @lightningPointLightRadius.setter
    def lightningPointLightRadius(self, arg: float, /) -> None: ...

    @property
    def lightningPointLightIntensity(self) -> float: ...

    @lightningPointLightIntensity.setter
    def lightningPointLightIntensity(self, arg: float, /) -> None: ...

    @property
    def thunderDelaySeconds(self) -> float: ...

    @thunderDelaySeconds.setter
    def thunderDelaySeconds(self, arg: float, /) -> None: ...

    @property
    def precipitationParticlesActive(self) -> bool: ...

    @precipitationParticlesActive.setter
    def precipitationParticlesActive(self, arg: bool, /) -> None: ...

    @property
    def snowParticlesActive(self) -> bool: ...

    @snowParticlesActive.setter
    def snowParticlesActive(self, arg: bool, /) -> None: ...

    @property
    def rainParticlesActive(self) -> bool: ...

    @rainParticlesActive.setter
    def rainParticlesActive(self, arg: bool, /) -> None: ...

    @property
    def precipitationParticleBudget(self) -> int: ...

    @precipitationParticleBudget.setter
    def precipitationParticleBudget(self, arg: int, /) -> None: ...

    @property
    def precipitationParticlesRendered(self) -> int: ...

    @precipitationParticlesRendered.setter
    def precipitationParticlesRendered(self, arg: int, /) -> None: ...

    @property
    def precipitationParticleSizePixels(self) -> float: ...

    @precipitationParticleSizePixels.setter
    def precipitationParticleSizePixels(self, arg: float, /) -> None: ...

    @property
    def precipitationParticleDepthMeters(self) -> float: ...

    @precipitationParticleDepthMeters.setter
    def precipitationParticleDepthMeters(self, arg: float, /) -> None: ...

    @property
    def precipitationParticleAlpha(self) -> float: ...

    @precipitationParticleAlpha.setter
    def precipitationParticleAlpha(self, arg: float, /) -> None: ...

    @property
    def precipitationParticleColor(self) -> numpy.ndarray: ...

    @precipitationParticleColor.setter
    def precipitationParticleColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def rainSplashParticlesActive(self) -> bool: ...

    @rainSplashParticlesActive.setter
    def rainSplashParticlesActive(self, arg: bool, /) -> None: ...

    @property
    def rainSplashParticlesRendered(self) -> int: ...

    @rainSplashParticlesRendered.setter
    def rainSplashParticlesRendered(self, arg: int, /) -> None: ...

    @property
    def rainSplashParticleSizePixels(self) -> float: ...

    @rainSplashParticleSizePixels.setter
    def rainSplashParticleSizePixels(self, arg: float, /) -> None: ...

    @property
    def rainSplashParticleAlpha(self) -> float: ...

    @rainSplashParticleAlpha.setter
    def rainSplashParticleAlpha(self, arg: float, /) -> None: ...

    @property
    def rainSplashParticleColor(self) -> numpy.ndarray: ...

    @rainSplashParticleColor.setter
    def rainSplashParticleColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lensDropletsActive(self) -> bool: ...

    @lensDropletsActive.setter
    def lensDropletsActive(self, arg: bool, /) -> None: ...

    @property
    def lensDropletsRendered(self) -> int: ...

    @lensDropletsRendered.setter
    def lensDropletsRendered(self, arg: int, /) -> None: ...

    @property
    def lensDropletStrength(self) -> float: ...

    @lensDropletStrength.setter
    def lensDropletStrength(self, arg: float, /) -> None: ...

    @property
    def lensDropletMaxSizePixels(self) -> float: ...

    @lensDropletMaxSizePixels.setter
    def lensDropletMaxSizePixels(self, arg: float, /) -> None: ...

    @property
    def lensDropletAlpha(self) -> float: ...

    @lensDropletAlpha.setter
    def lensDropletAlpha(self, arg: float, /) -> None: ...

    @property
    def weatherTextureBytes(self) -> int: ...

    @weatherTextureBytes.setter
    def weatherTextureBytes(self, arg: int, /) -> None: ...

    @property
    def transitionActive(self) -> bool: ...

    @transitionActive.setter
    def transitionActive(self, arg: bool, /) -> None: ...

    @property
    def transitionAlpha(self) -> float: ...

    @transitionAlpha.setter
    def transitionAlpha(self, arg: float, /) -> None: ...

    @property
    def skyEnvironmentValid(self) -> bool: ...

    @skyEnvironmentValid.setter
    def skyEnvironmentValid(self, arg: bool, /) -> None: ...

    @property
    def skyEnvironmentMap(self) -> int: ...

    @skyEnvironmentMap.setter
    def skyEnvironmentMap(self, arg: int, /) -> None: ...

    @property
    def skyIrradianceMap(self) -> int: ...

    @skyIrradianceMap.setter
    def skyIrradianceMap(self, arg: int, /) -> None: ...

    @property
    def skyEnvironmentFaceSize(self) -> int: ...

    @skyEnvironmentFaceSize.setter
    def skyEnvironmentFaceSize(self, arg: int, /) -> None: ...

    @property
    def skyIrradianceFaceSize(self) -> int: ...

    @skyIrradianceFaceSize.setter
    def skyIrradianceFaceSize(self, arg: int, /) -> None: ...

    @property
    def skyEnvironmentGenerationCount(self) -> int: ...

    @skyEnvironmentGenerationCount.setter
    def skyEnvironmentGenerationCount(self, arg: int, /) -> None: ...

class WeatherPreset(enum.Enum):
    Clear = 0

    Hazy = 1

    Overcast = 2

    Fog = 3

    Rain = 4

    HeavyRain = 5

    Snow = 6

    Storm = 7

    NightClear = 8

    NightRain = 9

    Custom = 10

class WeatherQuality(enum.Enum):
    Low = 0

    Medium = 1

    High = 2

    Ultra = 3

class WeatherSettings:
    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def preset(self) -> WeatherPreset: ...

    @preset.setter
    def preset(self, arg: WeatherPreset, /) -> None: ...

    @property
    def quality(self) -> WeatherQuality: ...

    @quality.setter
    def quality(self, arg: WeatherQuality, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def timeOfDayHours(self) -> float: ...

    @timeOfDayHours.setter
    def timeOfDayHours(self, arg: float, /) -> None: ...

    @property
    def latitude(self) -> float: ...

    @latitude.setter
    def latitude(self, arg: float, /) -> None: ...

    @property
    def longitude(self) -> float: ...

    @longitude.setter
    def longitude(self, arg: float, /) -> None: ...

    @property
    def automaticUtcOffset(self) -> bool: ...

    @automaticUtcOffset.setter
    def automaticUtcOffset(self, arg: bool, /) -> None: ...

    @property
    def utcOffsetHours(self) -> float: ...

    @utcOffsetHours.setter
    def utcOffsetHours(self, arg: float, /) -> None: ...

    @property
    def year(self) -> int: ...

    @year.setter
    def year(self, arg: int, /) -> None: ...

    @property
    def month(self) -> int: ...

    @month.setter
    def month(self, arg: int, /) -> None: ...

    @property
    def day(self) -> int: ...

    @day.setter
    def day(self, arg: int, /) -> None: ...

    @property
    def windDirection(self) -> numpy.ndarray: ...

    @windDirection.setter
    def windDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def windSpeed(self) -> float: ...

    @windSpeed.setter
    def windSpeed(self, arg: float, /) -> None: ...

    @property
    def transitionSeconds(self) -> float: ...

    @transitionSeconds.setter
    def transitionSeconds(self, arg: float, /) -> None: ...

    @property
    def affectSensors(self) -> bool: ...

    @affectSensors.setter
    def affectSensors(self, arg: bool, /) -> None: ...

    @property
    def cloudCoverage(self) -> float: ...

    @cloudCoverage.setter
    def cloudCoverage(self, arg: float, /) -> None: ...

    @property
    def cloudDensity(self) -> float: ...

    @cloudDensity.setter
    def cloudDensity(self, arg: float, /) -> None: ...

    @property
    def precipitationRate(self) -> float: ...

    @precipitationRate.setter
    def precipitationRate(self, arg: float, /) -> None: ...

    @property
    def rainOcclusionStrength(self) -> float: ...

    @rainOcclusionStrength.setter
    def rainOcclusionStrength(self, arg: float, /) -> None: ...

    @property
    def fogDensity(self) -> float: ...

    @fogDensity.setter
    def fogDensity(self, arg: float, /) -> None: ...

    @property
    def visibilityMeters(self) -> float: ...

    @visibilityMeters.setter
    def visibilityMeters(self, arg: float, /) -> None: ...

    @property
    def fogColor(self) -> numpy.ndarray: ...

    @fogColor.setter
    def fogColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def fogAnisotropy(self) -> float: ...

    @fogAnisotropy.setter
    def fogAnisotropy(self, arg: float, /) -> None: ...

    @property
    def humidity(self) -> float: ...

    @humidity.setter
    def humidity(self, arg: float, /) -> None: ...

    @property
    def wetness(self) -> float: ...

    @wetness.setter
    def wetness(self, arg: float, /) -> None: ...

    @property
    def wetnessAccumulationEnabled(self) -> bool: ...

    @wetnessAccumulationEnabled.setter
    def wetnessAccumulationEnabled(self, arg: bool, /) -> None: ...

    @property
    def wetnessAccumulationRate(self) -> float: ...

    @wetnessAccumulationRate.setter
    def wetnessAccumulationRate(self, arg: float, /) -> None: ...

    @property
    def wetnessDryingRate(self) -> float: ...

    @wetnessDryingRate.setter
    def wetnessDryingRate(self, arg: float, /) -> None: ...

    @property
    def snowCoverage(self) -> float: ...

    @snowCoverage.setter
    def snowCoverage(self, arg: float, /) -> None: ...

    @property
    def lightningRate(self) -> float: ...

    @lightningRate.setter
    def lightningRate(self, arg: float, /) -> None: ...

    @property
    def lightningLocalPointLightEnabled(self) -> bool: ...

    @lightningLocalPointLightEnabled.setter
    def lightningLocalPointLightEnabled(self, arg: bool, /) -> None: ...

    @property
    def lightningLocalPointDistance(self) -> float: ...

    @lightningLocalPointDistance.setter
    def lightningLocalPointDistance(self, arg: float, /) -> None: ...

    @property
    def lightningLocalPointRadius(self) -> float: ...

    @lightningLocalPointRadius.setter
    def lightningLocalPointRadius(self, arg: float, /) -> None: ...

    @property
    def lightningLocalPointIntensity(self) -> float: ...

    @lightningLocalPointIntensity.setter
    def lightningLocalPointIntensity(self, arg: float, /) -> None: ...

    @property
    def thunderSpeedOfSound(self) -> float: ...

    @thunderSpeedOfSound.setter
    def thunderSpeedOfSound(self, arg: float, /) -> None: ...

    @property
    def airTurbidity(self) -> float: ...

    @airTurbidity.setter
    def airTurbidity(self, arg: float, /) -> None: ...

    @property
    def groundAlbedo(self) -> float: ...

    @groundAlbedo.setter
    def groundAlbedo(self, arg: float, /) -> None: ...

    @property
    def useExplicitSunAngles(self) -> bool: ...

    @useExplicitSunAngles.setter
    def useExplicitSunAngles(self, arg: bool, /) -> None: ...

    @property
    def sunAzimuthDegrees(self) -> float: ...

    @sunAzimuthDegrees.setter
    def sunAzimuthDegrees(self, arg: float, /) -> None: ...

    @property
    def sunElevationDegrees(self) -> float: ...

    @sunElevationDegrees.setter
    def sunElevationDegrees(self, arg: float, /) -> None: ...

    @property
    def sunDiskSize(self) -> float: ...

    @sunDiskSize.setter
    def sunDiskSize(self, arg: float, /) -> None: ...

    @property
    def moonDiskSize(self) -> float: ...

    @moonDiskSize.setter
    def moonDiskSize(self, arg: float, /) -> None: ...

    @property
    def cloudAltitudeMeters(self) -> float: ...

    @cloudAltitudeMeters.setter
    def cloudAltitudeMeters(self, arg: float, /) -> None: ...

    @property
    def cloudThicknessMeters(self) -> float: ...

    @cloudThicknessMeters.setter
    def cloudThicknessMeters(self, arg: float, /) -> None: ...

    @property
    def cloudShadowStrength(self) -> float: ...

    @cloudShadowStrength.setter
    def cloudShadowStrength(self, arg: float, /) -> None: ...

    @property
    def cloudScale(self) -> float: ...

    @cloudScale.setter
    def cloudScale(self, arg: float, /) -> None: ...

    @property
    def cloudAnimationSpeed(self) -> float: ...

    @cloudAnimationSpeed.setter
    def cloudAnimationSpeed(self, arg: float, /) -> None: ...

    @property
    def lensDropletsEnabled(self) -> bool: ...

    @lensDropletsEnabled.setter
    def lensDropletsEnabled(self, arg: bool, /) -> None: ...

    @property
    def lensDropletStrength(self) -> float: ...

    @lensDropletStrength.setter
    def lensDropletStrength(self, arg: float, /) -> None: ...

class WeatherSkyEnvironment:
    def __init__(self) -> None: ...

    @property
    def environmentMap(self) -> int: ...

    @environmentMap.setter
    def environmentMap(self, arg: int, /) -> None: ...

    @property
    def irradianceMap(self) -> int: ...

    @irradianceMap.setter
    def irradianceMap(self, arg: int, /) -> None: ...

    @property
    def environmentFaceSize(self) -> int: ...

    @environmentFaceSize.setter
    def environmentFaceSize(self, arg: int, /) -> None: ...

    @property
    def irradianceFaceSize(self) -> int: ...

    @irradianceFaceSize.setter
    def irradianceFaceSize(self, arg: int, /) -> None: ...

    @property
    def generationCount(self) -> int: ...

    @generationCount.setter
    def generationCount(self, arg: int, /) -> None: ...

    @property
    def valid(self) -> bool: ...

    @valid.setter
    def valid(self, arg: bool, /) -> None: ...

class CaptureFilteredReflectionProbeOptions:
    def __init__(self) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def strength(self) -> float: ...

    @strength.setter
    def strength(self, arg: float, /) -> None: ...

    @property
    def capture(self) -> RayraiWindow.ReflectionProbeCaptureSettings | None: ...

    @capture.setter
    def capture(self, arg: RayraiWindow.ReflectionProbeCaptureSettings, /) -> None: ...

    @property
    def filter(self) -> RayraiWindow.ReflectionProbeFilterSettings | None: ...

    @filter.setter
    def filter(self, arg: RayraiWindow.ReflectionProbeFilterSettings, /) -> None: ...

class ConservativeOcclusionCullingDiagnostics:
    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def used(self) -> bool: ...

    @used.setter
    def used(self, arg: bool, /) -> None: ...

    @property
    def supported(self) -> bool: ...

    @supported.setter
    def supported(self, arg: bool, /) -> None: ...

    @property
    def hizEnabled(self) -> bool: ...

    @hizEnabled.setter
    def hizEnabled(self, arg: bool, /) -> None: ...

    @property
    def hizUsed(self) -> bool: ...

    @hizUsed.setter
    def hizUsed(self, arg: bool, /) -> None: ...

    @property
    def gpuHizEnabled(self) -> bool: ...

    @gpuHizEnabled.setter
    def gpuHizEnabled(self, arg: bool, /) -> None: ...

    @property
    def gpuHizUsed(self) -> bool: ...

    @gpuHizUsed.setter
    def gpuHizUsed(self, arg: bool, /) -> None: ...

    @property
    def candidates(self) -> int: ...

    @candidates.setter
    def candidates(self, arg: int, /) -> None: ...

    @property
    def queries(self) -> int: ...

    @queries.setter
    def queries(self, arg: int, /) -> None: ...

    @property
    def occluded(self) -> int: ...

    @occluded.setter
    def occluded(self, arg: int, /) -> None: ...

    @property
    def drawn(self) -> int: ...

    @drawn.setter
    def drawn(self, arg: int, /) -> None: ...

    @property
    def hizTests(self) -> int: ...

    @hizTests.setter
    def hizTests(self, arg: int, /) -> None: ...

    @property
    def hizOccluded(self) -> int: ...

    @hizOccluded.setter
    def hizOccluded(self, arg: int, /) -> None: ...

    @property
    def hizDepthReadbacks(self) -> int: ...

    @hizDepthReadbacks.setter
    def hizDepthReadbacks(self, arg: int, /) -> None: ...

    @property
    def hizMipLevels(self) -> int: ...

    @hizMipLevels.setter
    def hizMipLevels(self, arg: int, /) -> None: ...

    @property
    def gpuHizDrawCalls(self) -> int: ...

    @gpuHizDrawCalls.setter
    def gpuHizDrawCalls(self, arg: int, /) -> None: ...

    @property
    def gpuHizDepthBuilds(self) -> int: ...

    @gpuHizDepthBuilds.setter
    def gpuHizDepthBuilds(self, arg: int, /) -> None: ...

    @property
    def gpuHizComputeDepthBuilds(self) -> int: ...

    @gpuHizComputeDepthBuilds.setter
    def gpuHizComputeDepthBuilds(self, arg: int, /) -> None: ...

    @property
    def gpuHizRasterDepthBuilds(self) -> int: ...

    @gpuHizRasterDepthBuilds.setter
    def gpuHizRasterDepthBuilds(self, arg: int, /) -> None: ...

    @property
    def gpuHizMipLevels(self) -> int: ...

    @gpuHizMipLevels.setter
    def gpuHizMipLevels(self, arg: int, /) -> None: ...

    @property
    def fallbackCode(self) -> int: ...

    @fallbackCode.setter
    def fallbackCode(self, arg: int, /) -> None: ...

    @property
    def hizFallbackCode(self) -> int: ...

    @hizFallbackCode.setter
    def hizFallbackCode(self, arg: int, /) -> None: ...

    @property
    def gpuHizFallbackCode(self) -> int: ...

    @gpuHizFallbackCode.setter
    def gpuHizFallbackCode(self, arg: int, /) -> None: ...

class DebugPassCapture:
    def __init__(self) -> None: ...

    @property
    def fileName(self) -> str: ...

    @fileName.setter
    def fileName(self, arg: str, /) -> None: ...

    @property
    def label(self) -> str: ...

    @label.setter
    def label(self, arg: str, /) -> None: ...

    @property
    def pbrDebugOutput(self) -> int: ...

    @pbrDebugOutput.setter
    def pbrDebugOutput(self, arg: int, /) -> None: ...

    @property
    def postProcessDebugMode(self) -> PostProcessDebugMode: ...

    @postProcessDebugMode.setter
    def postProcessDebugMode(self, arg: PostProcessDebugMode, /) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def rgba(self) -> list[int]: ...

    @rgba.setter
    def rgba(self, arg: Sequence[int], /) -> None: ...

class DirectionalShadowCascade:
    def __init__(self) -> None: ...

    @property
    def index(self) -> int: ...

    @index.setter
    def index(self, arg: int, /) -> None: ...

    @property
    def nearPlane(self) -> float: ...

    @nearPlane.setter
    def nearPlane(self, arg: float, /) -> None: ...

    @property
    def farPlane(self) -> float: ...

    @farPlane.setter
    def farPlane(self, arg: float, /) -> None: ...

    @property
    def splitDepth(self) -> float: ...

    @splitDepth.setter
    def splitDepth(self, arg: float, /) -> None: ...

    @property
    def texelWorldSize(self) -> float: ...

    @texelWorldSize.setter
    def texelWorldSize(self, arg: float, /) -> None: ...

    @property
    def center(self) -> numpy.ndarray: ...

    @center.setter
    def center(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lightSpaceMin(self) -> numpy.ndarray: ...

    @lightSpaceMin.setter
    def lightSpaceMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lightSpaceMax(self) -> numpy.ndarray: ...

    @lightSpaceMax.setter
    def lightSpaceMax(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lightSpaceHalfExtents(self) -> numpy.ndarray: ...

    @lightSpaceHalfExtents.setter
    def lightSpaceHalfExtents(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def lightSpaceMatrix(self) -> numpy.ndarray: ...

    @lightSpaceMatrix.setter
    def lightSpaceMatrix(self, arg: numpy.ndarray, /) -> None: ...

class DirectionalShadowCascadePlan:
    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def requestedCascadeCount(self) -> int: ...

    @requestedCascadeCount.setter
    def requestedCascadeCount(self, arg: int, /) -> None: ...

    @property
    def cascadeCount(self) -> int: ...

    @cascadeCount.setter
    def cascadeCount(self, arg: int, /) -> None: ...

    @property
    def nearPlane(self) -> float: ...

    @nearPlane.setter
    def nearPlane(self, arg: float, /) -> None: ...

    @property
    def farPlane(self) -> float: ...

    @farPlane.setter
    def farPlane(self, arg: float, /) -> None: ...

    @property
    def maxDistance(self) -> float: ...

    @maxDistance.setter
    def maxDistance(self, arg: float, /) -> None: ...

    @property
    def shadowResolution(self) -> float: ...

    @shadowResolution.setter
    def shadowResolution(self, arg: float, /) -> None: ...

    @property
    def stableTexelSnap(self) -> bool: ...

    @stableTexelSnap.setter
    def stableTexelSnap(self, arg: bool, /) -> None: ...

    @property
    def splitOverridesEnabled(self) -> bool: ...

    @splitOverridesEnabled.setter
    def splitOverridesEnabled(self, arg: bool, /) -> None: ...

    @property
    def splitOverrides(self) -> list[float]: ...

    @splitOverrides.setter
    def splitOverrides(self, arg: Sequence[float], /) -> None: ...

    @property
    def cascades(self) -> list[RayraiWindow.DirectionalShadowCascade]: ...

    @cascades.setter
    def cascades(self, arg: Sequence[RayraiWindow.DirectionalShadowCascade], /) -> None: ...

class DynamicQualityRecommendation:
    def __init__(self) -> None: ...

    @property
    def changed(self) -> bool: ...

    @changed.setter
    def changed(self, arg: bool, /) -> None: ...

    @property
    def fillRateBound(self) -> bool: ...

    @fillRateBound.setter
    def fillRateBound(self, arg: bool, /) -> None: ...

    @property
    def shadowBound(self) -> bool: ...

    @shadowBound.setter
    def shadowBound(self, arg: bool, /) -> None: ...

    @property
    def postprocessBound(self) -> bool: ...

    @postprocessBound.setter
    def postprocessBound(self, arg: bool, /) -> None: ...

    @property
    def totalGpuMs(self) -> float: ...

    @totalGpuMs.setter
    def totalGpuMs(self, arg: float, /) -> None: ...

    @property
    def shadowGpuMs(self) -> float: ...

    @shadowGpuMs.setter
    def shadowGpuMs(self, arg: float, /) -> None: ...

    @property
    def sceneGpuMs(self) -> float: ...

    @sceneGpuMs.setter
    def sceneGpuMs(self, arg: float, /) -> None: ...

    @property
    def postprocessGpuMs(self) -> float: ...

    @postprocessGpuMs.setter
    def postprocessGpuMs(self, arg: float, /) -> None: ...

    @property
    def targetFrameMs(self) -> float: ...

    @targetFrameMs.setter
    def targetFrameMs(self, arg: float, /) -> None: ...

    @property
    def recommendedRenderScale(self) -> float: ...

    @recommendedRenderScale.setter
    def recommendedRenderScale(self, arg: float, /) -> None: ...

    @property
    def recommendedUpdateShadowsEveryFrame(self) -> bool: ...

    @recommendedUpdateShadowsEveryFrame.setter
    def recommendedUpdateShadowsEveryFrame(self, arg: bool, /) -> None: ...

    @property
    def recommendedBloomQuality(self) -> int: ...

    @recommendedBloomQuality.setter
    def recommendedBloomQuality(self, arg: int, /) -> None: ...

    @property
    def recommendedScreenSpaceAoSamples(self) -> int: ...

    @recommendedScreenSpaceAoSamples.setter
    def recommendedScreenSpaceAoSamples(self, arg: int, /) -> None: ...

    @property
    def recommendedScreenSpaceAoDenoiseEnabled(self) -> bool: ...

    @recommendedScreenSpaceAoDenoiseEnabled.setter
    def recommendedScreenSpaceAoDenoiseEnabled(self, arg: bool, /) -> None: ...

    @property
    def recommendedSettings(self) -> RenderQualitySettings: ...

    @recommendedSettings.setter
    def recommendedSettings(self, arg: RenderQualitySettings, /) -> None: ...

    @property
    def reason(self) -> str: ...

    @reason.setter
    def reason(self, arg: str, /) -> None: ...

class EnvironmentBackgroundDiagnostics:
    def __init__(self) -> None: ...

    @property
    def drawn(self) -> bool: ...

    @drawn.setter
    def drawn(self, arg: bool, /) -> None: ...

    @property
    def deferred(self) -> bool: ...

    @deferred.setter
    def deferred(self, arg: bool, /) -> None: ...

    @property
    def depthTested(self) -> bool: ...

    @depthTested.setter
    def depthTested(self, arg: bool, /) -> None: ...

class EnvironmentSidecarSettings:
    def __init__(self) -> None: ...

    @property
    def hasEnvironmentIntensity(self) -> bool: ...

    @hasEnvironmentIntensity.setter
    def hasEnvironmentIntensity(self, arg: bool, /) -> None: ...

    @property
    def environmentIntensity(self) -> float: ...

    @environmentIntensity.setter
    def environmentIntensity(self, arg: float, /) -> None: ...

    @property
    def hasEnvironmentRotation(self) -> bool: ...

    @hasEnvironmentRotation.setter
    def hasEnvironmentRotation(self, arg: bool, /) -> None: ...

    @property
    def environmentRotation(self) -> float: ...

    @environmentRotation.setter
    def environmentRotation(self, arg: float, /) -> None: ...

    @property
    def hasSkyTint(self) -> bool: ...

    @hasSkyTint.setter
    def hasSkyTint(self, arg: bool, /) -> None: ...

    @property
    def skyTint(self) -> numpy.ndarray: ...

    @skyTint.setter
    def skyTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def hasHorizonTint(self) -> bool: ...

    @hasHorizonTint.setter
    def hasHorizonTint(self, arg: bool, /) -> None: ...

    @property
    def horizonTint(self) -> numpy.ndarray: ...

    @horizonTint.setter
    def horizonTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def hasGroundTint(self) -> bool: ...

    @hasGroundTint.setter
    def hasGroundTint(self, arg: bool, /) -> None: ...

    @property
    def groundTint(self) -> numpy.ndarray: ...

    @groundTint.setter
    def groundTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def hasHorizonStrength(self) -> bool: ...

    @hasHorizonStrength.setter
    def hasHorizonStrength(self, arg: bool, /) -> None: ...

    @property
    def horizonStrength(self) -> float: ...

    @horizonStrength.setter
    def horizonStrength(self, arg: float, /) -> None: ...

    @property
    def hasBackgroundTint(self) -> bool: ...

    @hasBackgroundTint.setter
    def hasBackgroundTint(self, arg: bool, /) -> None: ...

    @property
    def backgroundTint(self) -> numpy.ndarray: ...

    @backgroundTint.setter
    def backgroundTint(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def hasBackgroundExposureScale(self) -> bool: ...

    @hasBackgroundExposureScale.setter
    def hasBackgroundExposureScale(self, arg: bool, /) -> None: ...

    @property
    def backgroundExposureScale(self) -> float: ...

    @backgroundExposureScale.setter
    def backgroundExposureScale(self, arg: float, /) -> None: ...

    @property
    def hasBackgroundMapPath(self) -> bool: ...

    @hasBackgroundMapPath.setter
    def hasBackgroundMapPath(self, arg: bool, /) -> None: ...

    @property
    def backgroundMapPath(self) -> str: ...

    @backgroundMapPath.setter
    def backgroundMapPath(self, arg: str, /) -> None: ...

    @property
    def hasBackgroundMapExposure(self) -> bool: ...

    @hasBackgroundMapExposure.setter
    def hasBackgroundMapExposure(self, arg: bool, /) -> None: ...

    @property
    def backgroundMapExposure(self) -> float: ...

    @backgroundMapExposure.setter
    def backgroundMapExposure(self, arg: float, /) -> None: ...

    @property
    def hasProceduralSkyEnabled(self) -> bool: ...

    @hasProceduralSkyEnabled.setter
    def hasProceduralSkyEnabled(self, arg: bool, /) -> None: ...

    @property
    def proceduralSkyEnabled(self) -> bool: ...

    @proceduralSkyEnabled.setter
    def proceduralSkyEnabled(self, arg: bool, /) -> None: ...

    @property
    def hasProceduralSkySunStrength(self) -> bool: ...

    @hasProceduralSkySunStrength.setter
    def hasProceduralSkySunStrength(self, arg: bool, /) -> None: ...

    @property
    def proceduralSkySunStrength(self) -> float: ...

    @proceduralSkySunStrength.setter
    def proceduralSkySunStrength(self, arg: float, /) -> None: ...

    @property
    def hasProceduralSkySunSize(self) -> bool: ...

    @hasProceduralSkySunSize.setter
    def hasProceduralSkySunSize(self, arg: bool, /) -> None: ...

    @property
    def proceduralSkySunSize(self) -> float: ...

    @proceduralSkySunSize.setter
    def proceduralSkySunSize(self, arg: float, /) -> None: ...

    @property
    def hasSunDirection(self) -> bool: ...

    @hasSunDirection.setter
    def hasSunDirection(self, arg: bool, /) -> None: ...

    @property
    def sunDirection(self) -> numpy.ndarray: ...

    @sunDirection.setter
    def sunDirection(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def hasSunColor(self) -> bool: ...

    @hasSunColor.setter
    def hasSunColor(self, arg: bool, /) -> None: ...

    @property
    def sunColor(self) -> numpy.ndarray: ...

    @sunColor.setter
    def sunColor(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def hasSunIntensity(self) -> bool: ...

    @hasSunIntensity.setter
    def hasSunIntensity(self, arg: bool, /) -> None: ...

    @property
    def sunIntensity(self) -> float: ...

    @sunIntensity.setter
    def sunIntensity(self, arg: float, /) -> None: ...

    @property
    def hasExposure(self) -> bool: ...

    @hasExposure.setter
    def hasExposure(self, arg: bool, /) -> None: ...

    @property
    def exposure(self) -> float: ...

    @exposure.setter
    def exposure(self, arg: float, /) -> None: ...

    @property
    def hasBloomEnabled(self) -> bool: ...

    @hasBloomEnabled.setter
    def hasBloomEnabled(self, arg: bool, /) -> None: ...

    @property
    def bloomEnabled(self) -> bool: ...

    @bloomEnabled.setter
    def bloomEnabled(self, arg: bool, /) -> None: ...

    @property
    def hasBloomThreshold(self) -> bool: ...

    @hasBloomThreshold.setter
    def hasBloomThreshold(self, arg: bool, /) -> None: ...

    @property
    def bloomThreshold(self) -> float: ...

    @bloomThreshold.setter
    def bloomThreshold(self, arg: float, /) -> None: ...

    @property
    def hasBloomStrength(self) -> bool: ...

    @hasBloomStrength.setter
    def hasBloomStrength(self, arg: bool, /) -> None: ...

    @property
    def bloomStrength(self) -> float: ...

    @bloomStrength.setter
    def bloomStrength(self, arg: float, /) -> None: ...

    @property
    def hasBloomRadius(self) -> bool: ...

    @hasBloomRadius.setter
    def hasBloomRadius(self, arg: bool, /) -> None: ...

    @property
    def bloomRadius(self) -> float: ...

    @bloomRadius.setter
    def bloomRadius(self, arg: float, /) -> None: ...

    @property
    def hasBloomKnee(self) -> bool: ...

    @hasBloomKnee.setter
    def hasBloomKnee(self, arg: bool, /) -> None: ...

    @property
    def bloomKnee(self) -> float: ...

    @bloomKnee.setter
    def bloomKnee(self, arg: float, /) -> None: ...

    @property
    def hasBloomQuality(self) -> bool: ...

    @hasBloomQuality.setter
    def hasBloomQuality(self, arg: bool, /) -> None: ...

    @property
    def bloomQuality(self) -> int: ...

    @bloomQuality.setter
    def bloomQuality(self, arg: int, /) -> None: ...

    @property
    def hasBloomSourceClamp(self) -> bool: ...

    @hasBloomSourceClamp.setter
    def hasBloomSourceClamp(self, arg: bool, /) -> None: ...

    @property
    def bloomSourceClamp(self) -> float: ...

    @bloomSourceClamp.setter
    def bloomSourceClamp(self, arg: float, /) -> None: ...

    @property
    def hasBloomAnamorphic(self) -> bool: ...

    @hasBloomAnamorphic.setter
    def hasBloomAnamorphic(self, arg: bool, /) -> None: ...

    @property
    def bloomAnamorphic(self) -> float: ...

    @bloomAnamorphic.setter
    def bloomAnamorphic(self, arg: float, /) -> None: ...

    @property
    def hasBloomDirtStrength(self) -> bool: ...

    @hasBloomDirtStrength.setter
    def hasBloomDirtStrength(self, arg: bool, /) -> None: ...

    @property
    def bloomDirtStrength(self) -> float: ...

    @bloomDirtStrength.setter
    def bloomDirtStrength(self, arg: float, /) -> None: ...

    @property
    def hasBloomDirtScale(self) -> bool: ...

    @hasBloomDirtScale.setter
    def hasBloomDirtScale(self, arg: bool, /) -> None: ...

    @property
    def bloomDirtScale(self) -> float: ...

    @bloomDirtScale.setter
    def bloomDirtScale(self, arg: float, /) -> None: ...

    @property
    def hasBloomBlendMode(self) -> bool: ...

    @hasBloomBlendMode.setter
    def hasBloomBlendMode(self, arg: bool, /) -> None: ...

    @property
    def bloomBlendMode(self) -> GlowBlendMode: ...

    @bloomBlendMode.setter
    def bloomBlendMode(self, arg: GlowBlendMode, /) -> None: ...

    @property
    def hasBloomMix(self) -> bool: ...

    @hasBloomMix.setter
    def hasBloomMix(self, arg: bool, /) -> None: ...

    @property
    def bloomMix(self) -> float: ...

    @bloomMix.setter
    def bloomMix(self, arg: float, /) -> None: ...

    @property
    def hasFogDensity(self) -> bool: ...

    @hasFogDensity.setter
    def hasFogDensity(self, arg: bool, /) -> None: ...

    @property
    def fogDensity(self) -> float: ...

    @fogDensity.setter
    def fogDensity(self, arg: float, /) -> None: ...

    @property
    def hasHeightFogEnabled(self) -> bool: ...

    @hasHeightFogEnabled.setter
    def hasHeightFogEnabled(self, arg: bool, /) -> None: ...

    @property
    def heightFogEnabled(self) -> bool: ...

    @heightFogEnabled.setter
    def heightFogEnabled(self, arg: bool, /) -> None: ...

    @property
    def hasHeightFogDensity(self) -> bool: ...

    @hasHeightFogDensity.setter
    def hasHeightFogDensity(self, arg: bool, /) -> None: ...

    @property
    def heightFogDensity(self) -> float: ...

    @heightFogDensity.setter
    def heightFogDensity(self, arg: float, /) -> None: ...

    @property
    def hasHeightFogBaseHeight(self) -> bool: ...

    @hasHeightFogBaseHeight.setter
    def hasHeightFogBaseHeight(self, arg: bool, /) -> None: ...

    @property
    def heightFogBaseHeight(self) -> float: ...

    @heightFogBaseHeight.setter
    def heightFogBaseHeight(self, arg: float, /) -> None: ...

    @property
    def hasHeightFogFalloff(self) -> bool: ...

    @hasHeightFogFalloff.setter
    def hasHeightFogFalloff(self, arg: bool, /) -> None: ...

    @property
    def heightFogFalloff(self) -> float: ...

    @heightFogFalloff.setter
    def heightFogFalloff(self, arg: float, /) -> None: ...

class ExposureRecommendation:
    def __init__(self) -> None: ...

    @property
    def valid(self) -> bool: ...

    @valid.setter
    def valid(self, arg: bool, /) -> None: ...

    @property
    def changed(self) -> bool: ...

    @changed.setter
    def changed(self, arg: bool, /) -> None: ...

    @property
    def underexposed(self) -> bool: ...

    @underexposed.setter
    def underexposed(self, arg: bool, /) -> None: ...

    @property
    def overexposed(self) -> bool: ...

    @overexposed.setter
    def overexposed(self, arg: bool, /) -> None: ...

    @property
    def clippingRisk(self) -> bool: ...

    @clippingRisk.setter
    def clippingRisk(self, arg: bool, /) -> None: ...

    @property
    def currentExposure(self) -> float: ...

    @currentExposure.setter
    def currentExposure(self, arg: float, /) -> None: ...

    @property
    def recommendedExposure(self) -> float: ...

    @recommendedExposure.setter
    def recommendedExposure(self, arg: float, /) -> None: ...

    @property
    def recommendedExposureScale(self) -> float: ...

    @recommendedExposureScale.setter
    def recommendedExposureScale(self, arg: float, /) -> None: ...

    @property
    def targetMedianLuminance(self) -> float: ...

    @targetMedianLuminance.setter
    def targetMedianLuminance(self, arg: float, /) -> None: ...

    @property
    def targetP95Luminance(self) -> float: ...

    @targetP95Luminance.setter
    def targetP95Luminance(self, arg: float, /) -> None: ...

    @property
    def minExposure(self) -> float: ...

    @minExposure.setter
    def minExposure(self, arg: float, /) -> None: ...

    @property
    def maxExposure(self) -> float: ...

    @maxExposure.setter
    def maxExposure(self, arg: float, /) -> None: ...

    @property
    def reason(self) -> str: ...

    @reason.setter
    def reason(self, arg: str, /) -> None: ...

class FoliageRenderDiagnostics:
    def __init__(self) -> None: ...

    @property
    def available(self) -> bool: ...

    @available.setter
    def available(self, arg: bool, /) -> None: ...

    @property
    def timingCaptureAvailable(self) -> bool: ...

    @timingCaptureAvailable.setter
    def timingCaptureAvailable(self, arg: bool, /) -> None: ...

    @property
    def sceneFoliageDrawCalls(self) -> int: ...

    @sceneFoliageDrawCalls.setter
    def sceneFoliageDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneFoliageOccludedDrawCalls(self) -> int: ...

    @sceneFoliageOccludedDrawCalls.setter
    def sceneFoliageOccludedDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneFoliageAuthoredInstanceVisits(self) -> int: ...

    @sceneFoliageAuthoredInstanceVisits.setter
    def sceneFoliageAuthoredInstanceVisits(self, arg: int, /) -> None: ...

    @property
    def sceneFoliageCandidateInstanceVisits(self) -> int: ...

    @sceneFoliageCandidateInstanceVisits.setter
    def sceneFoliageCandidateInstanceVisits(self, arg: int, /) -> None: ...

    @property
    def sceneFoliageRenderedInstanceVisits(self) -> int: ...

    @sceneFoliageRenderedInstanceVisits.setter
    def sceneFoliageRenderedInstanceVisits(self, arg: int, /) -> None: ...

    @property
    def shadowFoliageDrawCalls(self) -> int: ...

    @shadowFoliageDrawCalls.setter
    def shadowFoliageDrawCalls(self, arg: int, /) -> None: ...

    @property
    def shadowFoliageAuthoredInstanceVisits(self) -> int: ...

    @shadowFoliageAuthoredInstanceVisits.setter
    def shadowFoliageAuthoredInstanceVisits(self, arg: int, /) -> None: ...

    @property
    def shadowFoliageCandidateInstanceVisits(self) -> int: ...

    @shadowFoliageCandidateInstanceVisits.setter
    def shadowFoliageCandidateInstanceVisits(self, arg: int, /) -> None: ...

    @property
    def shadowFoliageRenderedInstanceVisits(self) -> int: ...

    @shadowFoliageRenderedInstanceVisits.setter
    def shadowFoliageRenderedInstanceVisits(self, arg: int, /) -> None: ...

    @property
    def sceneGpuCulledDrawCalls(self) -> int: ...

    @sceneGpuCulledDrawCalls.setter
    def sceneGpuCulledDrawCalls(self, arg: int, /) -> None: ...

    @property
    def shadowGpuCulledDrawCalls(self) -> int: ...

    @shadowGpuCulledDrawCalls.setter
    def shadowGpuCulledDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneGpuCullSkippedDrawCalls(self) -> int: ...

    @sceneGpuCullSkippedDrawCalls.setter
    def sceneGpuCullSkippedDrawCalls(self, arg: int, /) -> None: ...

    @property
    def shadowGpuCullSkippedDrawCalls(self) -> int: ...

    @shadowGpuCullSkippedDrawCalls.setter
    def shadowGpuCullSkippedDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneVisibleInstanceFrameCacheHits(self) -> int: ...

    @sceneVisibleInstanceFrameCacheHits.setter
    def sceneVisibleInstanceFrameCacheHits(self, arg: int, /) -> None: ...

    @property
    def sceneVisibleInstanceFrameCacheDispatches(self) -> int: ...

    @sceneVisibleInstanceFrameCacheDispatches.setter
    def sceneVisibleInstanceFrameCacheDispatches(self, arg: int, /) -> None: ...

    @property
    def sceneVisibleInstanceFrameCacheFallbackDrawCalls(self) -> int: ...

    @sceneVisibleInstanceFrameCacheFallbackDrawCalls.setter
    def sceneVisibleInstanceFrameCacheFallbackDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneClusteredDrawCalls(self) -> int: ...

    @sceneClusteredDrawCalls.setter
    def sceneClusteredDrawCalls(self, arg: int, /) -> None: ...

    @property
    def shadowClusteredDrawCalls(self) -> int: ...

    @shadowClusteredDrawCalls.setter
    def shadowClusteredDrawCalls(self, arg: int, /) -> None: ...

    @property
    def impostorDrawCalls(self) -> int: ...

    @impostorDrawCalls.setter
    def impostorDrawCalls(self, arg: int, /) -> None: ...

    @property
    def shadowLodDrawCalls(self) -> int: ...

    @shadowLodDrawCalls.setter
    def shadowLodDrawCalls(self, arg: int, /) -> None: ...

    @property
    def persistentSourceBufferDrawCalls(self) -> int: ...

    @persistentSourceBufferDrawCalls.setter
    def persistentSourceBufferDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneMultiDrawIndirectDrawCalls(self) -> int: ...

    @sceneMultiDrawIndirectDrawCalls.setter
    def sceneMultiDrawIndirectDrawCalls(self, arg: int, /) -> None: ...

    @property
    def shadowMultiDrawIndirectDrawCalls(self) -> int: ...

    @shadowMultiDrawIndirectDrawCalls.setter
    def shadowMultiDrawIndirectDrawCalls(self, arg: int, /) -> None: ...

    @property
    def multiDrawIndirectCommandCount(self) -> int: ...

    @multiDrawIndirectCommandCount.setter
    def multiDrawIndirectCommandCount(self, arg: int, /) -> None: ...

    @property
    def multiDrawIndirectFallbackDrawCalls(self) -> int: ...

    @multiDrawIndirectFallbackDrawCalls.setter
    def multiDrawIndirectFallbackDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneAlphaDepthPrepassDrawCalls(self) -> int: ...

    @sceneAlphaDepthPrepassDrawCalls.setter
    def sceneAlphaDepthPrepassDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneAlphaDepthPrepassInstanceVisits(self) -> int: ...

    @sceneAlphaDepthPrepassInstanceVisits.setter
    def sceneAlphaDepthPrepassInstanceVisits(self, arg: int, /) -> None: ...

    @property
    def sceneAlphaDepthPrepassFallbackDrawCalls(self) -> int: ...

    @sceneAlphaDepthPrepassFallbackDrawCalls.setter
    def sceneAlphaDepthPrepassFallbackDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneFrontToBackOrderedInstancedDrawCalls(self) -> int: ...

    @sceneFrontToBackOrderedInstancedDrawCalls.setter
    def sceneFrontToBackOrderedInstancedDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneFrontToBackOrderedFoliageDrawCalls(self) -> int: ...

    @sceneFrontToBackOrderedFoliageDrawCalls.setter
    def sceneFrontToBackOrderedFoliageDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneFrontToBackOrderingFallbackDrawCalls(self) -> int: ...

    @sceneFrontToBackOrderingFallbackDrawCalls.setter
    def sceneFrontToBackOrderingFallbackDrawCalls(self, arg: int, /) -> None: ...

    @property
    def maxSourceTrianglesPerInstance(self) -> int: ...

    @maxSourceTrianglesPerInstance.setter
    def maxSourceTrianglesPerInstance(self, arg: int, /) -> None: ...

    @property
    def maxImpostorTrianglesPerInstance(self) -> int: ...

    @maxImpostorTrianglesPerInstance.setter
    def maxImpostorTrianglesPerInstance(self, arg: int, /) -> None: ...

    @property
    def conservativeOcclusionCandidates(self) -> int: ...

    @conservativeOcclusionCandidates.setter
    def conservativeOcclusionCandidates(self, arg: int, /) -> None: ...

    @property
    def conservativeOcclusionQueries(self) -> int: ...

    @conservativeOcclusionQueries.setter
    def conservativeOcclusionQueries(self, arg: int, /) -> None: ...

    @property
    def conservativeOcclusionOccluded(self) -> int: ...

    @conservativeOcclusionOccluded.setter
    def conservativeOcclusionOccluded(self, arg: int, /) -> None: ...

    @property
    def conservativeOcclusionDrawn(self) -> int: ...

    @conservativeOcclusionDrawn.setter
    def conservativeOcclusionDrawn(self, arg: int, /) -> None: ...

    @property
    def hizOcclusionTests(self) -> int: ...

    @hizOcclusionTests.setter
    def hizOcclusionTests(self, arg: int, /) -> None: ...

    @property
    def hizOcclusionOccluded(self) -> int: ...

    @hizOcclusionOccluded.setter
    def hizOcclusionOccluded(self, arg: int, /) -> None: ...

    @property
    def hizOcclusionDepthReadbacks(self) -> int: ...

    @hizOcclusionDepthReadbacks.setter
    def hizOcclusionDepthReadbacks(self, arg: int, /) -> None: ...

    @property
    def gpuHizOcclusionDrawCalls(self) -> int: ...

    @gpuHizOcclusionDrawCalls.setter
    def gpuHizOcclusionDrawCalls(self, arg: int, /) -> None: ...

    @property
    def gpuHizOcclusionDepthBuilds(self) -> int: ...

    @gpuHizOcclusionDepthBuilds.setter
    def gpuHizOcclusionDepthBuilds(self, arg: int, /) -> None: ...

    @property
    def gpuHizOcclusionComputeDepthBuilds(self) -> int: ...

    @gpuHizOcclusionComputeDepthBuilds.setter
    def gpuHizOcclusionComputeDepthBuilds(self, arg: int, /) -> None: ...

    @property
    def gpuHizOcclusionRasterDepthBuilds(self) -> int: ...

    @gpuHizOcclusionRasterDepthBuilds.setter
    def gpuHizOcclusionRasterDepthBuilds(self, arg: int, /) -> None: ...

    @property
    def sceneGpuOcclusionFilteredDrawCalls(self) -> int: ...

    @sceneGpuOcclusionFilteredDrawCalls.setter
    def sceneGpuOcclusionFilteredDrawCalls(self, arg: int, /) -> None: ...

    @property
    def sceneFoliageGpuMs(self) -> float: ...

    @sceneFoliageGpuMs.setter
    def sceneFoliageGpuMs(self, arg: float, /) -> None: ...

    @property
    def shadowFoliageGpuMs(self) -> float: ...

    @shadowFoliageGpuMs.setter
    def shadowFoliageGpuMs(self, arg: float, /) -> None: ...

class ImportVisualSceneOptions:
    def __init__(self) -> None: ...

    @property
    def importLights(self) -> bool: ...

    @importLights.setter
    def importLights(self, arg: bool, /) -> None: ...

    @property
    def lightIntensityScale(self) -> float: ...

    @lightIntensityScale.setter
    def lightIntensityScale(self, arg: float, /) -> None: ...

    @property
    def scale(self) -> numpy.ndarray: ...

    @scale.setter
    def scale(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def color(self) -> numpy.ndarray: ...

    @color.setter
    def color(self, arg: numpy.ndarray, /) -> None: ...

class MaterialTextureBudgetRecommendation:
    def __init__(self) -> None: ...

    @property
    def materialTextureReferences(self) -> int: ...

    @materialTextureReferences.setter
    def materialTextureReferences(self, arg: int, /) -> None: ...

    @property
    def uniqueMaterialTextures(self) -> int: ...

    @uniqueMaterialTextures.setter
    def uniqueMaterialTextures(self, arg: int, /) -> None: ...

    @property
    def materialTextureBytes(self) -> int: ...

    @materialTextureBytes.setter
    def materialTextureBytes(self, arg: int, /) -> None: ...

    @property
    def budgetBytes(self) -> int: ...

    @budgetBytes.setter
    def budgetBytes(self, arg: int, /) -> None: ...

    @property
    def overBudget(self) -> bool: ...

    @overBudget.setter
    def overBudget(self, arg: bool, /) -> None: ...

    @property
    def changed(self) -> bool: ...

    @changed.setter
    def changed(self, arg: bool, /) -> None: ...

    @property
    def recommendedTextureAnisotropy(self) -> float: ...

    @recommendedTextureAnisotropy.setter
    def recommendedTextureAnisotropy(self, arg: float, /) -> None: ...

    @property
    def recommendedTextureMipLodBias(self) -> float: ...

    @recommendedTextureMipLodBias.setter
    def recommendedTextureMipLodBias(self, arg: float, /) -> None: ...

    @property
    def recommendedPbrEnvironmentMipLodBias(self) -> float: ...

    @recommendedPbrEnvironmentMipLodBias.setter
    def recommendedPbrEnvironmentMipLodBias(self, arg: float, /) -> None: ...

    @property
    def recommendedReflectionProbeFilteringEnabled(self) -> bool: ...

    @recommendedReflectionProbeFilteringEnabled.setter
    def recommendedReflectionProbeFilteringEnabled(self, arg: bool, /) -> None: ...

    @property
    def recommendedSettings(self) -> RenderQualitySettings: ...

    @recommendedSettings.setter
    def recommendedSettings(self, arg: RenderQualitySettings, /) -> None: ...

    @property
    def reason(self) -> str: ...

    @reason.setter
    def reason(self, arg: str, /) -> None: ...

class ReflectionProbe:
    def __init__(self) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def environmentMap(self) -> int: ...

    @environmentMap.setter
    def environmentMap(self, arg: int, /) -> None: ...

    @property
    def irradianceMap(self) -> int: ...

    @irradianceMap.setter
    def irradianceMap(self, arg: int, /) -> None: ...

    @property
    def prefilteredEnvironmentMap(self) -> int: ...

    @prefilteredEnvironmentMap.setter
    def prefilteredEnvironmentMap(self, arg: int, /) -> None: ...

    @property
    def brdfLut(self) -> int: ...

    @brdfLut.setter
    def brdfLut(self, arg: int, /) -> None: ...

    @property
    def strength(self) -> float: ...

    @strength.setter
    def strength(self, arg: float, /) -> None: ...

    @property
    def boxProjection(self) -> bool: ...

    @boxProjection.setter
    def boxProjection(self, arg: bool, /) -> None: ...

    @property
    def boxMin(self) -> numpy.ndarray: ...

    @boxMin.setter
    def boxMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def boxMax(self) -> numpy.ndarray: ...

    @boxMax.setter
    def boxMax(self, arg: numpy.ndarray, /) -> None: ...

class ReflectionProbeBlend:
    def __init__(self) -> None: ...

    @property
    def valid(self) -> bool: ...

    @valid.setter
    def valid(self, arg: bool, /) -> None: ...

    @property
    def primaryIndex(self) -> int: ...

    @primaryIndex.setter
    def primaryIndex(self, arg: int, /) -> None: ...

    @property
    def secondaryIndex(self) -> int: ...

    @secondaryIndex.setter
    def secondaryIndex(self, arg: int, /) -> None: ...

    @property
    def primaryWeight(self) -> float: ...

    @primaryWeight.setter
    def primaryWeight(self, arg: float, /) -> None: ...

    @property
    def secondaryWeight(self) -> float: ...

    @secondaryWeight.setter
    def secondaryWeight(self, arg: float, /) -> None: ...

    @property
    def strength(self) -> float: ...

    @strength.setter
    def strength(self, arg: float, /) -> None: ...

    @property
    def primaryBoxProjection(self) -> bool: ...

    @primaryBoxProjection.setter
    def primaryBoxProjection(self, arg: bool, /) -> None: ...

    @property
    def secondaryBoxProjection(self) -> bool: ...

    @secondaryBoxProjection.setter
    def secondaryBoxProjection(self, arg: bool, /) -> None: ...

    @property
    def primaryProbePosition(self) -> numpy.ndarray: ...

    @primaryProbePosition.setter
    def primaryProbePosition(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def secondaryProbePosition(self) -> numpy.ndarray: ...

    @secondaryProbePosition.setter
    def secondaryProbePosition(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def primaryBoxMin(self) -> numpy.ndarray: ...

    @primaryBoxMin.setter
    def primaryBoxMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def primaryBoxMax(self) -> numpy.ndarray: ...

    @primaryBoxMax.setter
    def primaryBoxMax(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def secondaryBoxMin(self) -> numpy.ndarray: ...

    @secondaryBoxMin.setter
    def secondaryBoxMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def secondaryBoxMax(self) -> numpy.ndarray: ...

    @secondaryBoxMax.setter
    def secondaryBoxMax(self, arg: numpy.ndarray, /) -> None: ...

class ReflectionProbeCaptureFrame:
    def __init__(self) -> None: ...

    @property
    def direction(self) -> numpy.ndarray: ...

    @direction.setter
    def direction(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def up(self) -> numpy.ndarray: ...

    @up.setter
    def up(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def right(self) -> numpy.ndarray: ...

    @right.setter
    def right(self, arg: numpy.ndarray, /) -> None: ...

class ReflectionProbeCaptureSettings:
    def __init__(self) -> None: ...

    @property
    def resolution(self) -> int: ...

    @resolution.setter
    def resolution(self, arg: int, /) -> None: ...

    @property
    def drawVisualizationObjects(self) -> bool: ...

    @drawVisualizationObjects.setter
    def drawVisualizationObjects(self, arg: bool, /) -> None: ...

    @property
    def drawCoordinateFrames(self) -> bool: ...

    @drawCoordinateFrames.setter
    def drawCoordinateFrames(self, arg: bool, /) -> None: ...

    @property
    def drawPointClouds(self) -> bool: ...

    @drawPointClouds.setter
    def drawPointClouds(self, arg: bool, /) -> None: ...

    @property
    def doShadows(self) -> bool: ...

    @doShadows.setter
    def doShadows(self, arg: bool, /) -> None: ...

    @property
    def includeEnvironmentBackground(self) -> bool: ...

    @includeEnvironmentBackground.setter
    def includeEnvironmentBackground(self, arg: bool, /) -> None: ...

    @property
    def environmentBackgroundMap(self) -> int: ...

    @environmentBackgroundMap.setter
    def environmentBackgroundMap(self, arg: int, /) -> None: ...

    @property
    def environmentBackgroundExposure(self) -> float: ...

    @environmentBackgroundExposure.setter
    def environmentBackgroundExposure(self, arg: float, /) -> None: ...

class ReflectionProbeDebugItem:
    def __init__(self) -> None: ...

    @property
    def index(self) -> int: ...

    @index.setter
    def index(self, arg: int, /) -> None: ...

    @property
    def selected(self) -> bool: ...

    @selected.setter
    def selected(self, arg: bool, /) -> None: ...

    @property
    def primary(self) -> bool: ...

    @primary.setter
    def primary(self, arg: bool, /) -> None: ...

    @property
    def secondary(self) -> bool: ...

    @secondary.setter
    def secondary(self, arg: bool, /) -> None: ...

    @property
    def rawWeight(self) -> float: ...

    @rawWeight.setter
    def rawWeight(self, arg: float, /) -> None: ...

    @property
    def normalizedWeight(self) -> float: ...

    @normalizedWeight.setter
    def normalizedWeight(self, arg: float, /) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def boxProjection(self) -> bool: ...

    @boxProjection.setter
    def boxProjection(self, arg: bool, /) -> None: ...

    @property
    def boxProjectionValid(self) -> bool: ...

    @boxProjectionValid.setter
    def boxProjectionValid(self, arg: bool, /) -> None: ...

    @property
    def boxMin(self) -> numpy.ndarray: ...

    @boxMin.setter
    def boxMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def boxMax(self) -> numpy.ndarray: ...

    @boxMax.setter
    def boxMax(self, arg: numpy.ndarray, /) -> None: ...

class ReflectionProbeDebugOverlaySummary:
    def __init__(self) -> None: ...

    @property
    def available(self) -> bool: ...

    @available.setter
    def available(self, arg: bool, /) -> None: ...

    @property
    def overlayEnabled(self) -> bool: ...

    @overlayEnabled.setter
    def overlayEnabled(self, arg: bool, /) -> None: ...

    @property
    def validBlend(self) -> bool: ...

    @validBlend.setter
    def validBlend(self, arg: bool, /) -> None: ...

    @property
    def registeredProbeCount(self) -> int: ...

    @registeredProbeCount.setter
    def registeredProbeCount(self, arg: int, /) -> None: ...

    @property
    def selectedProbeCount(self) -> int: ...

    @selectedProbeCount.setter
    def selectedProbeCount(self, arg: int, /) -> None: ...

    @property
    def boxProjectionProbeCount(self) -> int: ...

    @boxProjectionProbeCount.setter
    def boxProjectionProbeCount(self, arg: int, /) -> None: ...

    @property
    def validBoxProjectionProbeCount(self) -> int: ...

    @validBoxProjectionProbeCount.setter
    def validBoxProjectionProbeCount(self, arg: int, /) -> None: ...

    @property
    def primaryIndex(self) -> int: ...

    @primaryIndex.setter
    def primaryIndex(self, arg: int, /) -> None: ...

    @property
    def secondaryIndex(self) -> int: ...

    @secondaryIndex.setter
    def secondaryIndex(self, arg: int, /) -> None: ...

    @property
    def primaryWeight(self) -> float: ...

    @primaryWeight.setter
    def primaryWeight(self, arg: float, /) -> None: ...

    @property
    def secondaryWeight(self) -> float: ...

    @secondaryWeight.setter
    def secondaryWeight(self, arg: float, /) -> None: ...

    @property
    def strength(self) -> float: ...

    @strength.setter
    def strength(self, arg: float, /) -> None: ...

    @property
    def samplePosition(self) -> numpy.ndarray: ...

    @samplePosition.setter
    def samplePosition(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def primaryPosition(self) -> numpy.ndarray: ...

    @primaryPosition.setter
    def primaryPosition(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def secondaryPosition(self) -> numpy.ndarray: ...

    @secondaryPosition.setter
    def secondaryPosition(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def primaryBoxProjection(self) -> bool: ...

    @primaryBoxProjection.setter
    def primaryBoxProjection(self, arg: bool, /) -> None: ...

    @property
    def secondaryBoxProjection(self) -> bool: ...

    @secondaryBoxProjection.setter
    def secondaryBoxProjection(self, arg: bool, /) -> None: ...

class ReflectionProbeDebugView:
    def __init__(self) -> None: ...

    @property
    def valid(self) -> bool: ...

    @valid.setter
    def valid(self, arg: bool, /) -> None: ...

    @property
    def samplePosition(self) -> numpy.ndarray: ...

    @samplePosition.setter
    def samplePosition(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def blend(self) -> RayraiWindow.ReflectionProbeBlend: ...

    @blend.setter
    def blend(self, arg: RayraiWindow.ReflectionProbeBlend, /) -> None: ...

    @property
    def probes(self) -> list[RayraiWindow.ReflectionProbeDebugItem]: ...

    @probes.setter
    def probes(self, arg: Sequence[RayraiWindow.ReflectionProbeDebugItem], /) -> None: ...

class ReflectionProbeFilterSettings:
    def __init__(self) -> None: ...

    @property
    def irradianceResolution(self) -> int: ...

    @irradianceResolution.setter
    def irradianceResolution(self, arg: int, /) -> None: ...

    @property
    def irradianceSamples(self) -> int: ...

    @irradianceSamples.setter
    def irradianceSamples(self, arg: int, /) -> None: ...

    @property
    def prefilteredResolution(self) -> int: ...

    @prefilteredResolution.setter
    def prefilteredResolution(self, arg: int, /) -> None: ...

    @property
    def prefilteredMipLevels(self) -> int: ...

    @prefilteredMipLevels.setter
    def prefilteredMipLevels(self, arg: int, /) -> None: ...

    @property
    def prefilteredSamples(self) -> int: ...

    @prefilteredSamples.setter
    def prefilteredSamples(self, arg: int, /) -> None: ...

    @property
    def brdfLutSize(self) -> int: ...

    @brdfLutSize.setter
    def brdfLutSize(self, arg: int, /) -> None: ...

    @property
    def brdfLutSamples(self) -> int: ...

    @brdfLutSamples.setter
    def brdfLutSamples(self, arg: int, /) -> None: ...

class ReflectionProbePlacementSuggestion:
    def __init__(self) -> None: ...

    @property
    def valid(self) -> bool: ...

    @valid.setter
    def valid(self, arg: bool, /) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def boxProjection(self) -> bool: ...

    @boxProjection.setter
    def boxProjection(self, arg: bool, /) -> None: ...

    @property
    def boxMin(self) -> numpy.ndarray: ...

    @boxMin.setter
    def boxMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def boxMax(self) -> numpy.ndarray: ...

    @boxMax.setter
    def boxMax(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def boundedObjectCount(self) -> int: ...

    @boundedObjectCount.setter
    def boundedObjectCount(self, arg: int, /) -> None: ...

    @property
    def boundsDiagonal(self) -> float: ...

    @boundsDiagonal.setter
    def boundsDiagonal(self, arg: float, /) -> None: ...

class ReflectionProbeSidecarLoadResult:
    def __init__(self) -> None: ...

    @property
    def probes(self) -> list[RayraiWindow.ReflectionProbeSidecarProbe]: ...

    @probes.setter
    def probes(self, arg: Sequence[RayraiWindow.ReflectionProbeSidecarProbe], /) -> None: ...

    @property
    def settings(self) -> RayraiWindow.ReflectionProbeSidecarSettings: ...

    @settings.setter
    def settings(self, arg: RayraiWindow.ReflectionProbeSidecarSettings, /) -> None: ...

    @property
    def loadedSidecarPath(self) -> str: ...

    @loadedSidecarPath.setter
    def loadedSidecarPath(self, arg: str, /) -> None: ...

class ReflectionProbeSidecarProbe:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def position(self) -> numpy.ndarray: ...

    @position.setter
    def position(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def strength(self) -> float: ...

    @strength.setter
    def strength(self, arg: float, /) -> None: ...

    @property
    def boxProjection(self) -> bool: ...

    @boxProjection.setter
    def boxProjection(self, arg: bool, /) -> None: ...

    @property
    def boxMin(self) -> numpy.ndarray: ...

    @boxMin.setter
    def boxMin(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def boxMax(self) -> numpy.ndarray: ...

    @boxMax.setter
    def boxMax(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def capture(self) -> RayraiWindow.ReflectionProbeCaptureSettings: ...

    @capture.setter
    def capture(self, arg: RayraiWindow.ReflectionProbeCaptureSettings, /) -> None: ...

    @property
    def hasFilter(self) -> bool: ...

    @hasFilter.setter
    def hasFilter(self, arg: bool, /) -> None: ...

    @property
    def filter(self) -> RayraiWindow.ReflectionProbeFilterSettings: ...

    @filter.setter
    def filter(self, arg: RayraiWindow.ReflectionProbeFilterSettings, /) -> None: ...

class ReflectionProbeSidecarSettings:
    def __init__(self) -> None: ...

    @property
    def hasMaxReflectionProbesPerSelection(self) -> bool: ...

    @hasMaxReflectionProbesPerSelection.setter
    def hasMaxReflectionProbesPerSelection(self, arg: bool, /) -> None: ...

    @property
    def maxReflectionProbesPerSelection(self) -> int: ...

    @maxReflectionProbesPerSelection.setter
    def maxReflectionProbesPerSelection(self, arg: int, /) -> None: ...

    @property
    def hasMinReflectionProbeWeight(self) -> bool: ...

    @hasMinReflectionProbeWeight.setter
    def hasMinReflectionProbeWeight(self, arg: bool, /) -> None: ...

    @property
    def minReflectionProbeWeight(self) -> float: ...

    @minReflectionProbeWeight.setter
    def minReflectionProbeWeight(self, arg: float, /) -> None: ...

    @property
    def hasReflectionProbeFilteringEnabled(self) -> bool: ...

    @hasReflectionProbeFilteringEnabled.setter
    def hasReflectionProbeFilteringEnabled(self, arg: bool, /) -> None: ...

    @property
    def reflectionProbeFilteringEnabled(self) -> bool: ...

    @reflectionProbeFilteringEnabled.setter
    def reflectionProbeFilteringEnabled(self, arg: bool, /) -> None: ...

    @property
    def hasReflectionProbeFilter(self) -> bool: ...

    @hasReflectionProbeFilter.setter
    def hasReflectionProbeFilter(self, arg: bool, /) -> None: ...

    @property
    def reflectionProbeFilter(self) -> RayraiWindow.ReflectionProbeFilterSettings: ...

    @reflectionProbeFilter.setter
    def reflectionProbeFilter(self, arg: RayraiWindow.ReflectionProbeFilterSettings, /) -> None: ...

class RenderOverrides:
    def __init__(self) -> None: ...

    @property
    def mesh(self) -> Shader | None: ...

    @mesh.setter
    def mesh(self, value: Shader | None) -> None: ...

    @property
    def pointcloud(self) -> Shader | None: ...

    @pointcloud.setter
    def pointcloud(self, value: Shader | None) -> None: ...

    @property
    def picking(self) -> Shader | None: ...

    @picking.setter
    def picking(self, value: Shader | None) -> None: ...

    @property
    def post(self) -> Shader | None: ...

    @post.setter
    def post(self, value: Shader | None) -> None: ...

    @property
    def doShadows(self) -> object: ...

    @doShadows.setter
    def doShadows(self, arg: bool, /) -> None: ...

    @property
    def drawCoordinateFrames(self) -> object: ...

    @drawCoordinateFrames.setter
    def drawCoordinateFrames(self, arg: bool, /) -> None: ...

    @property
    def drawPointClouds(self) -> object: ...

    @drawPointClouds.setter
    def drawPointClouds(self, arg: bool, /) -> None: ...

    @property
    def drawVisualizationObjects(self) -> object: ...

    @drawVisualizationObjects.setter
    def drawVisualizationObjects(self, arg: bool, /) -> None: ...

    @property
    def drawWeatherEffects(self) -> object: ...

    @drawWeatherEffects.setter
    def drawWeatherEffects(self, arg: bool, /) -> None: ...

    @property
    def skipGroundObjects(self) -> object: ...

    @skipGroundObjects.setter
    def skipGroundObjects(self, arg: bool, /) -> None: ...

    @property
    def environmentBackgroundMap(self) -> object: ...

    @environmentBackgroundMap.setter
    def environmentBackgroundMap(self, arg: int, /) -> None: ...

    @property
    def environmentBackgroundExposure(self) -> object: ...

    @environmentBackgroundExposure.setter
    def environmentBackgroundExposure(self, arg: float, /) -> None: ...

    @property
    def postProcess(self) -> object: ...

    @postProcess.setter
    def postProcess(self, arg: bool, /) -> None: ...

    @property
    def allowViewerMsaa(self) -> object: ...

    @allowViewerMsaa.setter
    def allowViewerMsaa(self, arg: bool, /) -> None: ...

    @property
    def allowTemporalAa(self) -> object: ...

    @allowTemporalAa.setter
    def allowTemporalAa(self, arg: bool, /) -> None: ...

    @property
    def allowViewerUpscale(self) -> object: ...

    @allowViewerUpscale.setter
    def allowViewerUpscale(self, arg: bool, /) -> None: ...

    @property
    def fixedShadowCenterEnabled(self) -> object: ...

    @fixedShadowCenterEnabled.setter
    def fixedShadowCenterEnabled(self, arg: bool, /) -> None: ...

    @property
    def fixedShadowCenter(self) -> object: ...

    @fixedShadowCenter.setter
    def fixedShadowCenter(self, arg: numpy.ndarray, /) -> None: ...

class RenderPassAccounting:
    def __init__(self) -> None: ...

    @property
    def fastForwardPathEligible(self) -> bool: ...

    @fastForwardPathEligible.setter
    def fastForwardPathEligible(self, arg: bool, /) -> None: ...

    @property
    def postprocessActive(self) -> bool: ...

    @postprocessActive.setter
    def postprocessActive(self, arg: bool, /) -> None: ...

    @property
    def msaaActive(self) -> bool: ...

    @msaaActive.setter
    def msaaActive(self, arg: bool, /) -> None: ...

    @property
    def renderScaleActive(self) -> bool: ...

    @renderScaleActive.setter
    def renderScaleActive(self, arg: bool, /) -> None: ...

    @property
    def probeFilteringActive(self) -> bool: ...

    @probeFilteringActive.setter
    def probeFilteringActive(self, arg: bool, /) -> None: ...

    @property
    def localProbeSelectionActive(self) -> bool: ...

    @localProbeSelectionActive.setter
    def localProbeSelectionActive(self, arg: bool, /) -> None: ...

    @property
    def additionalShadowActive(self) -> bool: ...

    @additionalShadowActive.setter
    def additionalShadowActive(self, arg: bool, /) -> None: ...

    @property
    def mainShadowPassesPerFrame(self) -> int: ...

    @mainShadowPassesPerFrame.setter
    def mainShadowPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def additional2dShadowPassesPerFrame(self) -> int: ...

    @additional2dShadowPassesPerFrame.setter
    def additional2dShadowPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def additionalPointShadowFacePassesPerFrame(self) -> int: ...

    @additionalPointShadowFacePassesPerFrame.setter
    def additionalPointShadowFacePassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def shadowPassesPerFrame(self) -> int: ...

    @shadowPassesPerFrame.setter
    def shadowPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def sceneColorPassesPerFrame(self) -> int: ...

    @sceneColorPassesPerFrame.setter
    def sceneColorPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def planarReflectionPassesPerFrame(self) -> int: ...

    @planarReflectionPassesPerFrame.setter
    def planarReflectionPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def renderScaleResolvePassesPerFrame(self) -> int: ...

    @renderScaleResolvePassesPerFrame.setter
    def renderScaleResolvePassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def transparentPassesPerFrame(self) -> int: ...

    @transparentPassesPerFrame.setter
    def transparentPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def postprocessPassesPerFrame(self) -> int: ...

    @postprocessPassesPerFrame.setter
    def postprocessPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def bloomExtraPassesPerFrame(self) -> int: ...

    @bloomExtraPassesPerFrame.setter
    def bloomExtraPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def aoDenoisePassesPerFrame(self) -> int: ...

    @aoDenoisePassesPerFrame.setter
    def aoDenoisePassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def debugReadbackPassesPerFrame(self) -> int: ...

    @debugReadbackPassesPerFrame.setter
    def debugReadbackPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def totalEstimatedPassesPerFrame(self) -> int: ...

    @totalEstimatedPassesPerFrame.setter
    def totalEstimatedPassesPerFrame(self, arg: int, /) -> None: ...

    @property
    def activeAdditionalLights(self) -> int: ...

    @activeAdditionalLights.setter
    def activeAdditionalLights(self, arg: int, /) -> None: ...

    @property
    def activeAdditionalShadowLights(self) -> int: ...

    @activeAdditionalShadowLights.setter
    def activeAdditionalShadowLights(self, arg: int, /) -> None: ...

    @property
    def activeReflectionProbes(self) -> int: ...

    @activeReflectionProbes.setter
    def activeReflectionProbes(self, arg: int, /) -> None: ...

    @property
    def activeUltraFeatureCount(self) -> int: ...

    @activeUltraFeatureCount.setter
    def activeUltraFeatureCount(self, arg: int, /) -> None: ...

    @property
    def likelyBottleneck(self) -> str: ...

    @likelyBottleneck.setter
    def likelyBottleneck(self, arg: str, /) -> None: ...

class RenderPassTiming:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def gpuMs(self) -> float: ...

    @gpuMs.setter
    def gpuMs(self, arg: float, /) -> None: ...

class RgbaCalibrationMetrics:
    def __init__(self) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def patchCount(self) -> int: ...

    @patchCount.setter
    def patchCount(self, arg: int, /) -> None: ...

    @property
    def validPatchCount(self) -> int: ...

    @validPatchCount.setter
    def validPatchCount(self, arg: int, /) -> None: ...

    @property
    def averageRgbError(self) -> float: ...

    @averageRgbError.setter
    def averageRgbError(self, arg: float, /) -> None: ...

    @property
    def rmsRgbError(self) -> float: ...

    @rmsRgbError.setter
    def rmsRgbError(self, arg: float, /) -> None: ...

    @property
    def maxRgbError(self) -> float: ...

    @maxRgbError.setter
    def maxRgbError(self, arg: float, /) -> None: ...

    @property
    def averageLuminanceError(self) -> float: ...

    @averageLuminanceError.setter
    def averageLuminanceError(self, arg: float, /) -> None: ...

    @property
    def averageSaturationError(self) -> float: ...

    @averageSaturationError.setter
    def averageSaturationError(self, arg: float, /) -> None: ...

    @property
    def patches(self) -> list[RayraiWindow.RgbaCalibrationPatchResult]: ...

    @patches.setter
    def patches(self, arg: Sequence[RayraiWindow.RgbaCalibrationPatchResult], /) -> None: ...

class RgbaCalibrationPatch:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def x(self) -> int: ...

    @x.setter
    def x(self, arg: int, /) -> None: ...

    @property
    def y(self) -> int: ...

    @y.setter
    def y(self, arg: int, /) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def targetRgb(self) -> numpy.ndarray: ...

    @targetRgb.setter
    def targetRgb(self, arg: numpy.ndarray, /) -> None: ...

class RgbaCalibrationPatchResult:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def valid(self) -> bool: ...

    @valid.setter
    def valid(self, arg: bool, /) -> None: ...

    @property
    def sampleCount(self) -> int: ...

    @sampleCount.setter
    def sampleCount(self, arg: int, /) -> None: ...

    @property
    def measuredRgb(self) -> numpy.ndarray: ...

    @measuredRgb.setter
    def measuredRgb(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def targetRgb(self) -> numpy.ndarray: ...

    @targetRgb.setter
    def targetRgb(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def rgbError(self) -> float: ...

    @rgbError.setter
    def rgbError(self, arg: float, /) -> None: ...

    @property
    def maxChannelError(self) -> float: ...

    @maxChannelError.setter
    def maxChannelError(self, arg: float, /) -> None: ...

    @property
    def luminanceError(self) -> float: ...

    @luminanceError.setter
    def luminanceError(self, arg: float, /) -> None: ...

    @property
    def saturationError(self) -> float: ...

    @saturationError.setter
    def saturationError(self, arg: float, /) -> None: ...

class RgbaCalibrationRecommendation:
    def __init__(self) -> None: ...

    @property
    def valid(self) -> bool: ...

    @valid.setter
    def valid(self, arg: bool, /) -> None: ...

    @property
    def changed(self) -> bool: ...

    @changed.setter
    def changed(self, arg: bool, /) -> None: ...

    @property
    def neutralPatchCount(self) -> int: ...

    @neutralPatchCount.setter
    def neutralPatchCount(self, arg: int, /) -> None: ...

    @property
    def saturatedPatchCount(self) -> int: ...

    @saturatedPatchCount.setter
    def saturatedPatchCount(self, arg: int, /) -> None: ...

    @property
    def averageRgbErrorBefore(self) -> float: ...

    @averageRgbErrorBefore.setter
    def averageRgbErrorBefore(self, arg: float, /) -> None: ...

    @property
    def outputTransform(self) -> RayraiWindow.RgbaOutputTransform: ...

    @outputTransform.setter
    def outputTransform(self, arg: RayraiWindow.RgbaOutputTransform, /) -> None: ...

    @property
    def reason(self) -> str: ...

    @reason.setter
    def reason(self, arg: str, /) -> None: ...

class RgbaLuminanceHistogram:
    def __init__(self) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def pixelCount(self) -> int: ...

    @pixelCount.setter
    def pixelCount(self, arg: int, /) -> None: ...

    @property
    def binCount(self) -> int: ...

    @binCount.setter
    def binCount(self, arg: int, /) -> None: ...

    @property
    def peakBin(self) -> int: ...

    @peakBin.setter
    def peakBin(self, arg: int, /) -> None: ...

    @property
    def peakFraction(self) -> float: ...

    @peakFraction.setter
    def peakFraction(self, arg: float, /) -> None: ...

    @property
    def bins(self) -> list[int]: ...

    @bins.setter
    def bins(self, arg: Sequence[int], /) -> None: ...

class RgbaLuminanceMetrics:
    def __init__(self) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def pixelCount(self) -> int: ...

    @pixelCount.setter
    def pixelCount(self, arg: int, /) -> None: ...

    @property
    def averageLuminance(self) -> float: ...

    @averageLuminance.setter
    def averageLuminance(self, arg: float, /) -> None: ...

    @property
    def medianLuminance(self) -> float: ...

    @medianLuminance.setter
    def medianLuminance(self, arg: float, /) -> None: ...

    @property
    def p95Luminance(self) -> float: ...

    @p95Luminance.setter
    def p95Luminance(self, arg: float, /) -> None: ...

    @property
    def p99Luminance(self) -> float: ...

    @p99Luminance.setter
    def p99Luminance(self, arg: float, /) -> None: ...

    @property
    def minLuminance(self) -> float: ...

    @minLuminance.setter
    def minLuminance(self, arg: float, /) -> None: ...

    @property
    def maxLuminance(self) -> float: ...

    @maxLuminance.setter
    def maxLuminance(self, arg: float, /) -> None: ...

    @property
    def averageSaturation(self) -> float: ...

    @averageSaturation.setter
    def averageSaturation(self, arg: float, /) -> None: ...

    @property
    def contrastP95ToMedian(self) -> float: ...

    @contrastP95ToMedian.setter
    def contrastP95ToMedian(self, arg: float, /) -> None: ...

    @property
    def clippedChannelFraction(self) -> float: ...

    @clippedChannelFraction.setter
    def clippedChannelFraction(self, arg: float, /) -> None: ...

class RgbaOutputTransform:
    def __init__(self) -> None: ...

    @property
    def exposure(self) -> float: ...

    @exposure.setter
    def exposure(self, arg: float, /) -> None: ...

    @property
    def whiteBalance(self) -> numpy.ndarray: ...

    @whiteBalance.setter
    def whiteBalance(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def saturation(self) -> float: ...

    @saturation.setter
    def saturation(self, arg: float, /) -> None: ...

    @property
    def gamma(self) -> float: ...

    @gamma.setter
    def gamma(self, arg: float, /) -> None: ...

class SceneImportReport:
    def __init__(self) -> None: ...

    @property
    def visualLoaded(self) -> bool: ...

    @visualLoaded.setter
    def visualLoaded(self, arg: bool, /) -> None: ...

    @property
    def totalSceneLightCount(self) -> int: ...

    @totalSceneLightCount.setter
    def totalSceneLightCount(self, arg: int, /) -> None: ...

    @property
    def importedLightCount(self) -> int: ...

    @importedLightCount.setter
    def importedLightCount(self, arg: int, /) -> None: ...

    @property
    def droppedLightCount(self) -> int: ...

    @droppedLightCount.setter
    def droppedLightCount(self, arg: int, /) -> None: ...

    @property
    def shadowLightPromoted(self) -> bool: ...

    @shadowLightPromoted.setter
    def shadowLightPromoted(self, arg: bool, /) -> None: ...

    @property
    def remainingShadingLightCount(self) -> int: ...

    @remainingShadingLightCount.setter
    def remainingShadingLightCount(self, arg: int, /) -> None: ...

    @property
    def notes(self) -> str: ...

    @notes.setter
    def notes(self, arg: str, /) -> None: ...

class ShaderWarmupDiagnostics:
    def __init__(self) -> None: ...

    @property
    def shaderProgramCount(self) -> int: ...

    @shaderProgramCount.setter
    def shaderProgramCount(self, arg: int, /) -> None: ...

    @property
    def linkedProgramCount(self) -> int: ...

    @linkedProgramCount.setter
    def linkedProgramCount(self, arg: int, /) -> None: ...

    @property
    def totalCompileMs(self) -> float: ...

    @totalCompileMs.setter
    def totalCompileMs(self, arg: float, /) -> None: ...

    @property
    def slowestCompileMs(self) -> float: ...

    @slowestCompileMs.setter
    def slowestCompileMs(self, arg: float, /) -> None: ...

    @property
    def slowestShaderName(self) -> str: ...

    @slowestShaderName.setter
    def slowestShaderName(self, arg: str, /) -> None: ...

    @property
    def warmedUp(self) -> bool: ...

    @warmedUp.setter
    def warmedUp(self, arg: bool, /) -> None: ...

class ShadowAtlasPlan:
    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def overflow(self) -> bool: ...

    @overflow.setter
    def overflow(self, arg: bool, /) -> None: ...

    @property
    def pointShadowPolicy(self) -> PointShadowPolicy: ...

    @pointShadowPolicy.setter
    def pointShadowPolicy(self, arg: PointShadowPolicy, /) -> None: ...

    @property
    def pointShadowsUseStandaloneCubemaps(self) -> bool: ...

    @pointShadowsUseStandaloneCubemaps.setter
    def pointShadowsUseStandaloneCubemaps(self, arg: bool, /) -> None: ...

    @property
    def atlasResolution(self) -> int: ...

    @atlasResolution.setter
    def atlasResolution(self, arg: int, /) -> None: ...

    @property
    def padding(self) -> int: ...

    @padding.setter
    def padding(self, arg: int, /) -> None: ...

    @property
    def tileCount(self) -> int: ...

    @tileCount.setter
    def tileCount(self, arg: int, /) -> None: ...

    @property
    def nonPointTileCount(self) -> int: ...

    @nonPointTileCount.setter
    def nonPointTileCount(self, arg: int, /) -> None: ...

    @property
    def pointShadowFaceTileCount(self) -> int: ...

    @pointShadowFaceTileCount.setter
    def pointShadowFaceTileCount(self, arg: int, /) -> None: ...

    @property
    def occupancy(self) -> float: ...

    @occupancy.setter
    def occupancy(self, arg: float, /) -> None: ...

    @property
    def tiles(self) -> list[RayraiWindow.ShadowAtlasTile]: ...

    @tiles.setter
    def tiles(self, arg: Sequence[RayraiWindow.ShadowAtlasTile], /) -> None: ...

class ShadowAtlasTile:
    def __init__(self) -> None: ...

    @property
    def shadowIndex(self) -> int: ...

    @shadowIndex.setter
    def shadowIndex(self, arg: int, /) -> None: ...

    @property
    def sourceLightIndex(self) -> int: ...

    @sourceLightIndex.setter
    def sourceLightIndex(self, arg: int, /) -> None: ...

    @property
    def face(self) -> int: ...

    @face.setter
    def face(self, arg: int, /) -> None: ...

    @property
    def pointShadow(self) -> bool: ...

    @pointShadow.setter
    def pointShadow(self, arg: bool, /) -> None: ...

    @property
    def resolution(self) -> int: ...

    @resolution.setter
    def resolution(self, arg: int, /) -> None: ...

    @property
    def x(self) -> int: ...

    @x.setter
    def x(self, arg: int, /) -> None: ...

    @property
    def y(self) -> int: ...

    @y.setter
    def y(self, arg: int, /) -> None: ...

    @property
    def w(self) -> int: ...

    @w.setter
    def w(self, arg: int, /) -> None: ...

    @property
    def h(self) -> int: ...

    @h.setter
    def h(self, arg: int, /) -> None: ...

    @property
    def uvRect(self) -> numpy.ndarray: ...

    @uvRect.setter
    def uvRect(self, arg: numpy.ndarray, /) -> None: ...

class ShadowCacheDiagnostics:
    def __init__(self) -> None: ...

    @property
    def available(self) -> bool: ...

    @available.setter
    def available(self, arg: bool, /) -> None: ...

    @property
    def eligible(self) -> bool: ...

    @eligible.setter
    def eligible(self, arg: bool, /) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def fallback(self) -> bool: ...

    @fallback.setter
    def fallback(self, arg: bool, /) -> None: ...

    @property
    def shadowMapRenders(self) -> int: ...

    @shadowMapRenders.setter
    def shadowMapRenders(self, arg: int, /) -> None: ...

    @property
    def shadowMapReuses(self) -> int: ...

    @shadowMapReuses.setter
    def shadowMapReuses(self, arg: int, /) -> None: ...

    @property
    def fallbackFrames(self) -> int: ...

    @fallbackFrames.setter
    def fallbackFrames(self, arg: int, /) -> None: ...

    @property
    def perLightDirtyTrackingAvailable(self) -> bool: ...

    @perLightDirtyTrackingAvailable.setter
    def perLightDirtyTrackingAvailable(self, arg: bool, /) -> None: ...

    @property
    def perLightDirtyTrackingEnabled(self) -> bool: ...

    @perLightDirtyTrackingEnabled.setter
    def perLightDirtyTrackingEnabled(self, arg: bool, /) -> None: ...

    @property
    def perLightDirtyTrackingFallback(self) -> bool: ...

    @perLightDirtyTrackingFallback.setter
    def perLightDirtyTrackingFallback(self, arg: bool, /) -> None: ...

    @property
    def perLightDirtyTrackingFallbackFrames(self) -> int: ...

    @perLightDirtyTrackingFallbackFrames.setter
    def perLightDirtyTrackingFallbackFrames(self, arg: int, /) -> None: ...

    @property
    def additionalShadowMapRenders(self) -> int: ...

    @additionalShadowMapRenders.setter
    def additionalShadowMapRenders(self, arg: int, /) -> None: ...

    @property
    def additionalShadowMapReuses(self) -> int: ...

    @additionalShadowMapReuses.setter
    def additionalShadowMapReuses(self, arg: int, /) -> None: ...

    @property
    def additionalShadowDirtyLights(self) -> int: ...

    @additionalShadowDirtyLights.setter
    def additionalShadowDirtyLights(self, arg: int, /) -> None: ...

    @property
    def additionalShadowCleanLights(self) -> int: ...

    @additionalShadowCleanLights.setter
    def additionalShadowCleanLights(self, arg: int, /) -> None: ...

    @property
    def staticDynamicCompositionAvailable(self) -> bool: ...

    @staticDynamicCompositionAvailable.setter
    def staticDynamicCompositionAvailable(self, arg: bool, /) -> None: ...

    @property
    def staticDynamicCompositionEnabled(self) -> bool: ...

    @staticDynamicCompositionEnabled.setter
    def staticDynamicCompositionEnabled(self, arg: bool, /) -> None: ...

    @property
    def staticDynamicCompositionFallback(self) -> bool: ...

    @staticDynamicCompositionFallback.setter
    def staticDynamicCompositionFallback(self, arg: bool, /) -> None: ...

    @property
    def staticDynamicCompositionFallbackFrames(self) -> int: ...

    @staticDynamicCompositionFallbackFrames.setter
    def staticDynamicCompositionFallbackFrames(self, arg: int, /) -> None: ...

    @property
    def staticShadowMapRenders(self) -> int: ...

    @staticShadowMapRenders.setter
    def staticShadowMapRenders(self, arg: int, /) -> None: ...

    @property
    def staticShadowMapReuses(self) -> int: ...

    @staticShadowMapReuses.setter
    def staticShadowMapReuses(self, arg: int, /) -> None: ...

    @property
    def dynamicShadowMapRenders(self) -> int: ...

    @dynamicShadowMapRenders.setter
    def dynamicShadowMapRenders(self, arg: int, /) -> None: ...

    @property
    def staticDynamicCompositedLayers(self) -> int: ...

    @staticDynamicCompositedLayers.setter
    def staticDynamicCompositedLayers(self, arg: int, /) -> None: ...

    @property
    def additionalStaticShadowMapRenders(self) -> int: ...

    @additionalStaticShadowMapRenders.setter
    def additionalStaticShadowMapRenders(self, arg: int, /) -> None: ...

    @property
    def additionalStaticShadowMapReuses(self) -> int: ...

    @additionalStaticShadowMapReuses.setter
    def additionalStaticShadowMapReuses(self, arg: int, /) -> None: ...

    @property
    def additionalDynamicShadowMapRenders(self) -> int: ...

    @additionalDynamicShadowMapRenders.setter
    def additionalDynamicShadowMapRenders(self, arg: int, /) -> None: ...

    @property
    def additionalStaticDynamicCompositedTiles(self) -> int: ...

    @additionalStaticDynamicCompositedTiles.setter
    def additionalStaticDynamicCompositedTiles(self, arg: int, /) -> None: ...

    @property
    def staticShadowCasterCount(self) -> int: ...

    @staticShadowCasterCount.setter
    def staticShadowCasterCount(self, arg: int, /) -> None: ...

    @property
    def dynamicShadowCasterCount(self) -> int: ...

    @dynamicShadowCasterCount.setter
    def dynamicShadowCasterCount(self, arg: int, /) -> None: ...

    @property
    def receiverDepthFilteredMaps(self) -> int: ...

    @receiverDepthFilteredMaps.setter
    def receiverDepthFilteredMaps(self, arg: int, /) -> None: ...

class ShadowDebugOverlaySummary:
    def __init__(self) -> None: ...

    @property
    def available(self) -> bool: ...

    @available.setter
    def available(self, arg: bool, /) -> None: ...

    @property
    def overlayEnabled(self) -> bool: ...

    @overlayEnabled.setter
    def overlayEnabled(self, arg: bool, /) -> None: ...

    @property
    def shadowsEnabled(self) -> bool: ...

    @shadowsEnabled.setter
    def shadowsEnabled(self, arg: bool, /) -> None: ...

    @property
    def updateEveryFrame(self) -> bool: ...

    @updateEveryFrame.setter
    def updateEveryFrame(self, arg: bool, /) -> None: ...

    @property
    def contactHardening(self) -> bool: ...

    @contactHardening.setter
    def contactHardening(self, arg: bool, /) -> None: ...

    @property
    def cascadeRendered(self) -> bool: ...

    @cascadeRendered.setter
    def cascadeRendered(self, arg: bool, /) -> None: ...

    @property
    def atlasRendered(self) -> bool: ...

    @atlasRendered.setter
    def atlasRendered(self, arg: bool, /) -> None: ...

    @property
    def atlasOverflow(self) -> bool: ...

    @atlasOverflow.setter
    def atlasOverflow(self, arg: bool, /) -> None: ...

    @property
    def splitOverridesEnabled(self) -> bool: ...

    @splitOverridesEnabled.setter
    def splitOverridesEnabled(self, arg: bool, /) -> None: ...

    @property
    def cascadeCount(self) -> int: ...

    @cascadeCount.setter
    def cascadeCount(self, arg: int, /) -> None: ...

    @property
    def atlasTileCount(self) -> int: ...

    @atlasTileCount.setter
    def atlasTileCount(self, arg: int, /) -> None: ...

    @property
    def atlasNonPointTileCount(self) -> int: ...

    @atlasNonPointTileCount.setter
    def atlasNonPointTileCount(self, arg: int, /) -> None: ...

    @property
    def pointShadowFaceTileCount(self) -> int: ...

    @pointShadowFaceTileCount.setter
    def pointShadowFaceTileCount(self, arg: int, /) -> None: ...

    @property
    def selectedAdditionalShadowLights(self) -> int: ...

    @selectedAdditionalShadowLights.setter
    def selectedAdditionalShadowLights(self, arg: int, /) -> None: ...

    @property
    def droppedShadowCandidates(self) -> int: ...

    @droppedShadowCandidates.setter
    def droppedShadowCandidates(self, arg: int, /) -> None: ...

    @property
    def shadowBias(self) -> float: ...

    @shadowBias.setter
    def shadowBias(self, arg: float, /) -> None: ...

    @property
    def shadowNormalOffset(self) -> float: ...

    @shadowNormalOffset.setter
    def shadowNormalOffset(self, arg: float, /) -> None: ...

    @property
    def shadowPcfRadius(self) -> float: ...

    @shadowPcfRadius.setter
    def shadowPcfRadius(self, arg: float, /) -> None: ...

    @property
    def shadowStrength(self) -> float: ...

    @shadowStrength.setter
    def shadowStrength(self, arg: float, /) -> None: ...

    @property
    def cascadeLambda(self) -> float: ...

    @cascadeLambda.setter
    def cascadeLambda(self, arg: float, /) -> None: ...

    @property
    def cascadeMaxDistance(self) -> float: ...

    @cascadeMaxDistance.setter
    def cascadeMaxDistance(self, arg: float, /) -> None: ...

    @property
    def atlasOccupancy(self) -> float: ...

    @atlasOccupancy.setter
    def atlasOccupancy(self, arg: float, /) -> None: ...

    @property
    def cascadeSplits(self) -> list[float]: ...

    @cascadeSplits.setter
    def cascadeSplits(self, arg: Sequence[float], /) -> None: ...

    @property
    def cascadeSplitOverrides(self) -> list[float]: ...

    @cascadeSplitOverrides.setter
    def cascadeSplitOverrides(self, arg: Sequence[float], /) -> None: ...

    @property
    def cascadeMode(self) -> str: ...

    @cascadeMode.setter
    def cascadeMode(self, arg: str, /) -> None: ...

    @property
    def atlasMode(self) -> str: ...

    @atlasMode.setter
    def atlasMode(self, arg: str, /) -> None: ...

class SupersampledCapture:
    def __init__(self) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def scale(self) -> int: ...

    @scale.setter
    def scale(self, arg: int, /) -> None: ...

    @property
    def rgba(self) -> list[int]: ...

    @rgba.setter
    def rgba(self, arg: Sequence[int], /) -> None: ...

class SynchronousMeshLoadingScope:
    def __init__(self, viewer: RayraiWindow, window: RayraiWindow) -> None: ...

    def __enter__(self) -> RayraiWindow.SynchronousMeshLoadingScope: ...

    def __exit__(self, exc_type: object | None, exc_value: object | None, traceback: object | None) -> None: ...

class ThreadingMode(enum.Enum):
    SingleThread = 0

    MultiThread = 1

class TransparentDrawDebugView:
    def __init__(self) -> None: ...

    @property
    def sortTransparentInstances(self) -> bool: ...

    @sortTransparentInstances.setter
    def sortTransparentInstances(self, arg: bool, /) -> None: ...

    @property
    def weightedBlendedOit(self) -> bool: ...

    @weightedBlendedOit.setter
    def weightedBlendedOit(self, arg: bool, /) -> None: ...

    @property
    def weightedBlendedOitAvailable(self) -> bool: ...

    @weightedBlendedOitAvailable.setter
    def weightedBlendedOitAvailable(self, arg: bool, /) -> None: ...

    @property
    def transparentDepthPrepass(self) -> bool: ...

    @transparentDepthPrepass.setter
    def transparentDepthPrepass(self, arg: bool, /) -> None: ...

    @property
    def transparentShadowDithering(self) -> bool: ...

    @transparentShadowDithering.setter
    def transparentShadowDithering(self, arg: bool, /) -> None: ...

    @property
    def transparentShadowOpacityScale(self) -> float: ...

    @transparentShadowOpacityScale.setter
    def transparentShadowOpacityScale(self, arg: float, /) -> None: ...

    @property
    def screenSpaceRefraction(self) -> bool: ...

    @screenSpaceRefraction.setter
    def screenSpaceRefraction(self, arg: bool, /) -> None: ...

    @property
    def screenSpaceRefractionAvailable(self) -> bool: ...

    @screenSpaceRefractionAvailable.setter
    def screenSpaceRefractionAvailable(self, arg: bool, /) -> None: ...

    @property
    def screenSpaceRefractionStrength(self) -> float: ...

    @screenSpaceRefractionStrength.setter
    def screenSpaceRefractionStrength(self, arg: float, /) -> None: ...

    @property
    def screenSpaceRefractionMaxPixels(self) -> float: ...

    @screenSpaceRefractionMaxPixels.setter
    def screenSpaceRefractionMaxPixels(self, arg: float, /) -> None: ...

    @property
    def transparentFogAlphaWeighted(self) -> bool: ...

    @transparentFogAlphaWeighted.setter
    def transparentFogAlphaWeighted(self, arg: bool, /) -> None: ...

    @property
    def transparentFogStrength(self) -> float: ...

    @transparentFogStrength.setter
    def transparentFogStrength(self, arg: float, /) -> None: ...

    @property
    def alphaToCoverage(self) -> bool: ...

    @alphaToCoverage.setter
    def alphaToCoverage(self, arg: bool, /) -> None: ...

    @property
    def depthWrite(self) -> bool: ...

    @depthWrite.setter
    def depthWrite(self, arg: bool, /) -> None: ...

    @property
    def depthTest(self) -> bool: ...

    @depthTest.setter
    def depthTest(self, arg: bool, /) -> None: ...

    @property
    def overdrawTileSize(self) -> int: ...

    @overdrawTileSize.setter
    def overdrawTileSize(self, arg: int, /) -> None: ...

    @property
    def overdrawTileColumns(self) -> int: ...

    @overdrawTileColumns.setter
    def overdrawTileColumns(self, arg: int, /) -> None: ...

    @property
    def overdrawTileRows(self) -> int: ...

    @overdrawTileRows.setter
    def overdrawTileRows(self, arg: int, /) -> None: ...

    @property
    def overdrawCoveredTileCount(self) -> int: ...

    @overdrawCoveredTileCount.setter
    def overdrawCoveredTileCount(self, arg: int, /) -> None: ...

    @property
    def overdrawHotTileCount(self) -> int: ...

    @overdrawHotTileCount.setter
    def overdrawHotTileCount(self, arg: int, /) -> None: ...

    @property
    def maxEstimatedOverdraw(self) -> int: ...

    @maxEstimatedOverdraw.setter
    def maxEstimatedOverdraw(self, arg: int, /) -> None: ...

    @property
    def averageEstimatedOverdraw(self) -> float: ...

    @averageEstimatedOverdraw.setter
    def averageEstimatedOverdraw(self, arg: float, /) -> None: ...

    @property
    def overdrawTileCounts(self) -> list[int]: ...

    @overdrawTileCounts.setter
    def overdrawTileCounts(self, arg: Sequence[int], /) -> None: ...

    @property
    def worldTransparentCount(self) -> int: ...

    @worldTransparentCount.setter
    def worldTransparentCount(self, arg: int, /) -> None: ...

    @property
    def visualTransparentCount(self) -> int: ...

    @visualTransparentCount.setter
    def visualTransparentCount(self, arg: int, /) -> None: ...

    @property
    def instancedTransparentCount(self) -> int: ...

    @instancedTransparentCount.setter
    def instancedTransparentCount(self, arg: int, /) -> None: ...

    @property
    def totalTransparentCount(self) -> int: ...

    @totalTransparentCount.setter
    def totalTransparentCount(self, arg: int, /) -> None: ...

    @property
    def backToFrontOrder(self) -> list[RayraiWindow.TransparentDrawItem]: ...

    @backToFrontOrder.setter
    def backToFrontOrder(self, arg: Sequence[RayraiWindow.TransparentDrawItem], /) -> None: ...

class TransparentDrawItem:
    def __init__(self) -> None: ...

    @property
    def source(self) -> str: ...

    @source.setter
    def source(self, arg: str, /) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def center(self) -> numpy.ndarray: ...

    @center.setter
    def center(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def radius(self) -> float: ...

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def distanceSquared(self) -> float: ...

    @distanceSquared.setter
    def distanceSquared(self, arg: float, /) -> None: ...

    @property
    def sortOffset(self) -> float: ...

    @sortOffset.setter
    def sortOffset(self, arg: float, /) -> None: ...

    @property
    def renderPriority(self) -> int: ...

    @renderPriority.setter
    def renderPriority(self, arg: int, /) -> None: ...

    @property
    def detectable(self) -> bool: ...

    @detectable.setter
    def detectable(self, arg: bool, /) -> None: ...

    @property
    def sortUsesBoundsCenter(self) -> bool: ...

    @sortUsesBoundsCenter.setter
    def sortUsesBoundsCenter(self, arg: bool, /) -> None: ...

class VisualMeshBuilder:
    def __init__(self, viewer: RayraiWindow, renderer: RayraiWindow, name: str, meshFile: str) -> None: ...

    @overload
    def scale(self, s: numpy.ndarray) -> object:
        """
        /// Per-axis mesh scale (default 1). @param s Scale. @return *this.
        C++: VisualMeshBuilder &(const glm::dvec3 &)
        """

    @overload
    def scale(self, x: float, y: float, z: float) -> object:
        """
        /// Per-axis mesh scale. @param x X scale. @param y Y scale. @param z Z scale. @return *this.
        C++: VisualMeshBuilder &(double, double, double)
        """

    def color(self, c: numpy.ndarray) -> object:
        """
        /// RGBA color in [0, 1] (default white). @param c Color. @return *this.
        C++: VisualMeshBuilder &(const glm::vec4 &)
        """

    def position(self, p: numpy.ndarray) -> object:
        """
        /// World-space position. @param p Position in metres. @return *this.
        C++: VisualMeshBuilder &(const glm::vec3 &)
        """

    def orientation(self, q: numpy.ndarray) -> object:
        """
        /// World-space orientation. @param q Unit quaternion. @return *this.
        C++: VisualMeshBuilder &(const glm::quat &)
        """

    def twoSided(self, enabled: bool = True) -> object:
        """
        /// Disable back-face culling. @param enabled Flag. @return *this.
        C++: VisualMeshBuilder &(bool)
        """

    def pickable(self, enabled: bool = True) -> object:
        """
        /// Sets Visuals::setDetectable(), which controls picking and visibility in
            /// sensor-style captures. @param enabled Flag. @return *this.
        C++: VisualMeshBuilder &(bool)
        """

    def castsShadows(self, enabled: bool = True) -> object:
        """
        /// Whether the visual casts shadows. @param enabled Flag. @return *this.
        C++: VisualMeshBuilder &(bool)
        """

    def useMeshColor(self, enabled: bool) -> object:
        """
        /// Use the mesh file's own colors instead of color(). @param enabled Flag. @return *this.
        C++: VisualMeshBuilder &(bool)
        """

    @overload
    def material(self, m: Material) -> object:
        """
        /// Material override shared with the caller. @param m Material. @return *this.
        C++: VisualMeshBuilder &(std::shared_ptr<Material>)
        """

    @overload
    def material(self, m: Material) -> object:
        """
        /// Material override (copied). @param m Material. @return *this.
        C++: VisualMeshBuilder &(const Material &)
        """

    def pbrEnvironment(self, env: PbrEnvironment) -> object:
        """
        /// Image-based lighting maps for the visual. @param env Environment. @return *this.
        C++: VisualMeshBuilder &(const PbrEnvironment &)
        """

    def add(self) -> object:
        """
        /** @brief Create the visual, apply the deferred state, return the shared_ptr.
             *  @return The new visual (the name must be unique, `RSFATAL` otherwise). */
        C++: std::shared_ptr<Visuals> ()
        """

class WeatherSidecarSettings:
    def __init__(self) -> None: ...

    @property
    def hasWeather(self) -> bool: ...

    @hasWeather.setter
    def hasWeather(self, arg: bool, /) -> None: ...

    @property
    def weather(self) -> WeatherSettings: ...

    @weather.setter
    def weather(self, arg: WeatherSettings, /) -> None: ...

    @property
    def hasLocalFogVolumes(self) -> bool: ...

    @hasLocalFogVolumes.setter
    def hasLocalFogVolumes(self, arg: bool, /) -> None: ...

    @property
    def localFogVolumes(self) -> list[LocalFogVolume]: ...

    @localFogVolumes.setter
    def localFogVolumes(self, arg: Sequence[LocalFogVolume], /) -> None: ...

def envFlagEnabled(name: str) -> bool:
    """C++: bool (const char *)"""

def envIntValue(name: str, fallback: int = 0) -> int:
    """C++: int (const char *, int)"""

def accumulateIrradiance(probe: Sequence[numpy.ndarray], radiance: numpy.ndarray, direction: numpy.ndarray, weight: float) -> tuple:
    """
    C++: void (IrradianceProbe &, const glm::vec3 &, const glm::vec3 &, float)
    Returns a tuple containing the native result (when present) followed by output parameters.
    """

def deringIrradiance(probe: Sequence[numpy.ndarray]) -> tuple:
    """
    C++: void (IrradianceProbe &)
    Returns a tuple containing the native result (when present) followed by output parameters.
    """

@overload
def toGlm(v: "raisim::Mat<3ul, 1ul>") -> numpy.ndarray:
    """C++: glm::dvec3 (const raisim::Vec<3> &)"""

@overload
def toGlm(R: "raisim::Mat<3ul, 3ul>") -> numpy.ndarray:
    """C++: glm::dmat3 (const raisim::Mat<3, 3> &)"""

def GenMeshCube(width: float, height: float, length: float) -> list[OpenGLMesh]:
    """C++: MeshList (float, float, float)"""

def GenMeshPlane(width: float, height: float, uTiling: float = 1.0, vTiling: float = 1.0, subdivisions: int = 1) -> list[OpenGLMesh]:
    """C++: MeshList (float, float, float, float, int)"""

def GenMeshSphere(radius: float, rings: int, sectors: int) -> list[OpenGLMesh]:
    """C++: MeshList (float, int, int)"""

def GenMeshCylinder(radius: float, height: float, sectors: int) -> list[OpenGLMesh]:
    """C++: MeshList (float, float, int)"""

def GenMeshCapsule(radius: float, height: float, sectors: int) -> list[OpenGLMesh]:
    """C++: MeshList (float, float, int)"""

def HeightmapVisualSampleCountForAxis(sourceSamples: int, worldSizeMeters: float) -> int:
    """C++: int (int, float)"""

@overload
def HeightmapVisualSampleCounts(height: Sequence[float], samplesX: int, samplesY: int, sizeX: float, sizeY: float) -> numpy.ndarray:
    """C++: glm::ivec2 (const std::vector<double> &, int, int, float, float)"""

@overload
def HeightmapVisualSampleCounts(height: Sequence[float], samplesX: int, samplesY: int, sizeX: float, sizeY: float) -> numpy.ndarray:
    """C++: glm::ivec2 (const std::vector<float> &, int, int, float, float)"""

def BuildHeightmapTriangleIndices(samplesX: int, samplesY: int) -> list[int]:
    """C++: std::vector<unsigned int> (int, int)"""

def RemoveInvalidHeightmapTriangles(vertices: Sequence[Vertex], indices: Sequence[int]) -> tuple:
    """
    C++: void (std::vector<Vertex> &, std::vector<unsigned int> &)
    Returns a tuple containing the native result (when present) followed by output parameters.
    """

@overload
def SampleHeightmapTriangulated(height: Sequence[float], samplesX: int, samplesY: int, sourceX: float, sourceY: float) -> float:
    """C++: float (const std::vector<double> &, int, int, float, float)"""

@overload
def SampleHeightmapTriangulated(height: Sequence[float], samplesX: int, samplesY: int, sourceX: float, sourceY: float) -> float:
    """C++: float (const std::vector<float> &, int, int, float, float)"""

def FillHeightmapVerticesRaisim(hm: raisimpy.HeightMap | None) -> tuple:
    """
    C++: bool (raisim::HeightMap *, std::vector<Vertex> &)
    Returns a tuple containing the native result (when present) followed by output parameters.
    """

def FillHeightmapVertexRowsRaisim(hm: raisimpy.HeightMap | None, sourceMinY: int, sourceMaxY: int) -> tuple:
    """
    C++: bool (raisim::HeightMap *, size_t, size_t, std::vector<Vertex> &, int &, int &, int &)
    Returns a tuple containing the native result (when present) followed by output parameters.
    """

def HeightmapVisualRowsOfSourceRows(sourceRows: int, visualRows: int, sourceMinY: int, sourceMaxY: int) -> numpy.ndarray:
    """C++: glm::ivec2 (int, int, size_t, size_t)"""

def GenMeshHeightmapRaisim(hm: raisimpy.HeightMap | None) -> list[OpenGLMesh]:
    """C++: MeshList (raisim::HeightMap *)"""

def findAsMeshFile(resourceDir: str, file: str) -> str:
    """C++: std::string (const std::string &, const std::string &)"""

def parseColorLiteral(input: str) -> numpy.ndarray | None:
    """C++: std::optional<glm::vec4> (std::string_view)"""

def parseNamedColor(name: str) -> numpy.ndarray | None:
    """C++: std::optional<glm::vec4> (std::string_view)"""

def isHiddenAppearance(appearance: str) -> bool:
    """C++: bool (std::string_view)"""

def appearanceToGlmVec4(appearance: str) -> numpy.ndarray:
    """C++: glm::vec4 (const std::string &)"""

def generateGerstnerWaves(spectrum: WaterWaveSpectrum) -> list[GerstnerWave]:
    """C++: std::vector<GerstnerWave> (const WaterWaveSpectrum &)"""

def defaultWaterBody(preset: WaterPreset) -> WaterBody:
    """C++: WaterBody (WaterPreset)"""

def waterPresetFromName(name: str) -> WaterPreset | None:
    """C++: std::optional<WaterPreset> (std::string_view)"""

def waterBodyForFluidVolume(volume: raisimpy.FluidVolume, preset: WaterPreset) -> WaterBody:
    """C++: WaterBody (const raisim::FluidVolume &, WaterPreset)"""

def validateWaterBody(body: WaterBody) -> None:
    """C++: void (const WaterBody &)"""

def sampleWaterSurface(body: WaterBody, worldXY: numpy.ndarray, timeSeconds: float) -> WaterSurfaceSample:
    """
    C++: WaterSurfaceSample (const WaterBody &, const glm::dvec2 &, double)
    """

def maxWaterWaveHeight(body: WaterBody) -> float:
    """C++: float (const WaterBody &)"""

def rayraiProcessIsMultiThreaded() -> bool:
    """C++: bool ()"""

def meshPreprocessCacheEnabled() -> bool:
    """C++: bool ()"""

def resetMeshPreprocessCacheStats() -> None:
    """C++: void ()"""

def meshPreprocessCacheStatsSnapshot() -> MeshPreprocessCacheStats:
    """C++: MeshPreprocessCacheStats ()"""

def computeMeshPreprocessMetadata(meshes: Sequence[OpenGLMesh]) -> MeshPreprocessMetadata:
    """
    C++: MeshPreprocessMetadata (const std::vector<std::shared_ptr<OpenGLMesh>> &)
    """

def recordMeshPreprocessImportSources(canonicalKey: str, files: Sequence[assimp.ModelSourceFile]) -> None:
    """
    C++: void (const std::string &, const std::vector<assimp::ModelSourceFile> &)
    """

def loadMeshPreprocessMetadata(canonicalKey: str, metadata: MeshPreprocessMetadata) -> bool:
    """C++: bool (const std::string &, MeshPreprocessMetadata &)"""

def storeMeshPreprocessMetadata(canonicalKey: str, metadata: MeshPreprocessMetadata) -> bool:
    """C++: bool (const std::string &, const MeshPreprocessMetadata &)"""

def loadMeshPreprocessLods(canonicalKey: str, sourceMeshes: Sequence[OpenGLMesh], flavor: MeshPreprocessLodFlavor) -> RayraiGlobalAsset.MeshLodChain:
    """
    C++: std::shared_ptr<RayraiGlobalAsset::MeshLodChain> (const std::string &, const std::vector<std::shared_ptr<OpenGLMesh>> &, MeshPreprocessLodFlavor)
    """

def storeMeshPreprocessLods(canonicalKey: str, sourceMeshes: Sequence[OpenGLMesh], flavor: MeshPreprocessLodFlavor, chain: RayraiGlobalAsset.MeshLodChain) -> bool:
    """
    C++: bool (const std::string &, const std::vector<std::shared_ptr<OpenGLMesh>> &, MeshPreprocessLodFlavor, const RayraiGlobalAsset::MeshLodChain &)
    """

def warnMeshPreprocessCacheFallbackOnce(canonicalKey: str, code: int, detail: str = '') -> None:
    """C++: void (const std::string &, uint32_t, const std::string &)"""

class Font:
    pass

class TextureHandle:
    @property
    def id(self) -> int: ...

class ModelHandle:
    pass

@overload
def syncRaisimCameraPose(sensorCam: raisimpy.RGBCamera, renderCam: Camera) -> None: ...

@overload
def syncRaisimCameraPose(sensorCam: raisimpy.DepthCamera, renderCam: Camera) -> None: ...

def closeAll() -> None: ...
