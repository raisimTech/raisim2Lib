from collections.abc import Sequence

import numpy

import raisimpy_rayrai


class AsyncModelLoadResult:
    def __init__(self) -> None: ...

    @property
    def path(self) -> str: ...

    @path.setter
    def path(self, arg: str, /) -> None: ...

    @property
    def directory(self) -> str: ...

    @directory.setter
    def directory(self, arg: str, /) -> None: ...

    @property
    def error(self) -> str: ...

    @error.setter
    def error(self, arg: str, /) -> None: ...

    @property
    def unsupportedGltfExtensionsNote(self) -> str: ...

    @unsupportedGltfExtensionsNote.setter
    def unsupportedGltfExtensionsNote(self, arg: str, /) -> None: ...

    @property
    def gltfBaseColorTextureFallbacks(self) -> list[str]: ...

    @gltfBaseColorTextureFallbacks.setter
    def gltfBaseColorTextureFallbacks(self, arg: Sequence[str], /) -> None: ...

    @property
    def fallback(self) -> FallbackMaterial: ...

    @fallback.setter
    def fallback(self, arg: FallbackMaterial, /) -> None: ...

    @property
    def stlDiffuseAlphaDefaultsOpaque(self) -> bool: ...

    @stlDiffuseAlphaDefaultsOpaque.setter
    def stlDiffuseAlphaDefaultsOpaque(self, arg: bool, /) -> None: ...

    @property
    def sourceFiles(self) -> list[ModelSourceFile]: ...

    @sourceFiles.setter
    def sourceFiles(self, arg: Sequence[ModelSourceFile], /) -> None: ...

class FallbackMaterial:
    def __init__(self) -> None: ...

    @property
    def hasColor(self) -> bool: ...

    @hasColor.setter
    def hasColor(self, arg: bool, /) -> None: ...

    @property
    def color(self) -> numpy.ndarray: ...

    @color.setter
    def color(self, arg: numpy.ndarray, /) -> None: ...

    @property
    def diffuseTex(self) -> str: ...

    @diffuseTex.setter
    def diffuseTex(self, arg: str, /) -> None: ...

    @property
    def specularTex(self) -> str: ...

    @specularTex.setter
    def specularTex(self, arg: str, /) -> None: ...

class ModelSourceFile:
    def __init__(self) -> None: ...

    @property
    def path(self) -> str: ...

    @path.setter
    def path(self, arg: str, /) -> None: ...

    @property
    def exists(self) -> bool: ...

    @exists.setter
    def exists(self, arg: bool, /) -> None: ...

    @property
    def size(self) -> int: ...

    @size.setter
    def size(self, arg: int, /) -> None: ...

    @property
    def time(self) -> int: ...

    @time.setter
    def time(self, arg: int, /) -> None: ...

    @staticmethod
    def current(path: str) -> raisimpy_rayrai.assimp.ModelSourceFile:
        """
        /// @return @p path as it is now.
        C++: ModelSourceFile (const std::string &)
        """

    def changed(self) -> bool:
        """
        /// @return True if the file was created, removed, resized or modified since.
        C++: bool () const
        """

def isUsdModelFile(path: str) -> bool:
    """C++: bool (const std::string &)"""

def importedTexturesDecoded(imported: AsyncModelLoadResult) -> bool:
    """C++: bool (const AsyncModelLoadResult &)"""

def finishImportedModelScene(imported: AsyncModelLoadResult) -> list[raisimpy_rayrai.OpenGLMesh]:
    """
    C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> (AsyncModelLoadResult &)
    """

def prepareImportedModelScene(imported: AsyncModelLoadResult) -> list[raisimpy_rayrai.OpenGLMesh]:
    """
    C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> (AsyncModelLoadResult &)
    """

def prepareImportedModelGeometry(meshes: Sequence[raisimpy_rayrai.OpenGLMesh], path: str) -> tuple:
    """
    C++: void (std::vector<std::shared_ptr<OpenGLMesh>> &, const std::string &)
    Returns a tuple containing the native result (when present) followed by output parameters.
    """

def loadModel(path: str) -> list[raisimpy_rayrai.OpenGLMesh]:
    """
    C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> (const std::string &)
    """

def loadModelWithSourceFiles(path: str) -> tuple:
    """
    C++: std::shared_ptr<std::vector<std::shared_ptr<OpenGLMesh>>> (const std::string &, std::vector<ModelSourceFile> &)
    Returns a tuple containing the native result (when present) followed by output parameters.
    """

def loadSceneLights(path: str, intensityScale: float = 1.0) -> list[raisimpy_rayrai.AdditionalLight]:
    """C++: std::vector<AdditionalLight> (const std::string &, float)"""

class ModelLoadFuture:
    def done(self) -> bool: ...

    def result(self) -> AsyncModelLoadResult: ...

def loadModelSceneAsync(path: str) -> ModelLoadFuture: ...
