from collections.abc import Callable, Sequence
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

from . import (
    blast as blast,
    constraints as constraints,
    contact as contact,
    rayrai as rayrai
)


__license__: str = 'PROPRIETARY'

hasWrenchSensors: bool = True

hasSegmentation: bool = True

__authors__: list = ...

class Transformation:
    """Raisim homogeneous transformation."""

    def __init__(self) -> None: ...

    @property
    def rot(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @rot.setter
    def rot(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def pos(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @pos.setter
    def pos(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

class SparseJacobian:
    """Raisim sparse Jacobian (3 x n)."""

    def __init__(self) -> None:
        """Initialize an empty sparse Jacobian."""

    def resize(self, cols: int) -> None:
        """Resize the Jacobian."""

    @property
    def size(self) -> int: ...

    @property
    def capacity(self) -> int: ...

    @property
    def idx(self) -> list[int]: ...

    @idx.setter
    def idx(self, arg: Sequence[int], /) -> None: ...

    def toDense(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Return a dense (3 x n) numpy array."""

class ColorRGB:
    """RGB color with 8-bit channels."""

    def __init__(self) -> None: ...

    @property
    def r(self) -> int: ...

    @r.setter
    def r(self, arg: int, /) -> None: ...

    @property
    def g(self) -> int: ...

    @g.setter
    def g(self, arg: int, /) -> None: ...

    @property
    def b(self) -> int: ...

    @b.setter
    def b(self, arg: int, /) -> None: ...

class ColorRGBA:
    """RGBA color with 8-bit channels."""

    def __init__(self) -> None: ...

    @property
    def r(self) -> int: ...

    @r.setter
    def r(self, arg: int, /) -> None: ...

    @property
    def g(self) -> int: ...

    @g.setter
    def g(self, arg: int, /) -> None: ...

    @property
    def b(self) -> int: ...

    @b.setter
    def b(self, arg: int, /) -> None: ...

    @property
    def a(self) -> int: ...

    @a.setter
    def a(self, arg: int, /) -> None: ...

class MaterialPairProperties:
    """Raisim Material Pair Properties (friction and restitution)."""

    @overload
    def __init__(self) -> None:
        """Initialize the material pair properties."""

    @overload
    def __init__(self, friction: float, restitution: float, restitution_threshold: float) -> None:
        """
        Initialize the material pair properties.

        Args:
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            threshold (float): restitution threshold.
        """

    @overload
    def __init__(self, friction: float, restitution: float, restitution_threshold: float, static_friction: float, static_friction_velocity_threshold: float, rolling_friction: float = 0.0, spinning_friction: float = 0.0) -> None:
        """
        Initialize the material pair properties.

        Args:
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            threshold (float): restitution threshold.
            static_friction (float): coefficient of static friction.
            static_friction_velocity_threshold (float): If the relative velocity of two contact points is bigger than this value, then the dynamic coefficient of friction is applied. Otherwise, the coefficient of friction is interpolated between the static and dynamic one proportional to the relative velocity.
            rolling_friction (float): coefficient of rolling friction. Defaults to 0.0.
            spinning_friction (float): coefficient of spinning friction. Defaults to 0.0.
        """

    @property
    def c_f(self) -> float: ...

    @c_f.setter
    def c_f(self, arg: float, /) -> None: ...

    @property
    def c_r(self) -> float: ...

    @c_r.setter
    def c_r(self, arg: float, /) -> None: ...

    @property
    def r_th(self) -> float: ...

    @r_th.setter
    def r_th(self, arg: float, /) -> None: ...

    @property
    def c_static_f(self) -> float: ...

    @c_static_f.setter
    def c_static_f(self, arg: float, /) -> None: ...

    @property
    def v_static_speed(self) -> float: ...

    @v_static_speed.setter
    def v_static_speed(self, arg: float, /) -> None: ...

    @property
    def v_static_speed_inv(self) -> float: ...

    @v_static_speed_inv.setter
    def v_static_speed_inv(self, arg: float, /) -> None: ...

    @property
    def c_rolling_f(self) -> float: ...

    @c_rolling_f.setter
    def c_rolling_f(self, arg: float, /) -> None: ...

    @property
    def c_spinning_f(self) -> float: ...

    @c_spinning_f.setter
    def c_spinning_f(self, arg: float, /) -> None: ...

class MaterialManager:
    """Raisim Material Manager."""

    @overload
    def __init__(self) -> None:
        """Initialize the material pair manager."""

    @overload
    def __init__(self, xml_file: str) -> None:
        """
        Initialize the material manager by uploading the material data from a file.

        Args:
            xml_file (float): xml file.
        """

    @overload
    def setMaterialPairProp(self, material1: str, material2: str, friction: float, restitution: float, restitution_threshold: float, static_friction: float, static_friction_velocity_threshold: float) -> None:
        """
        Set the material pair properties (friction and restitution).

        Args:
            material1 (str): first material.
            material2 (str): second material.
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            restitution_threshold (float): restitution threshold.
            static friction (float): coefficient of static friction
            static friction velocity threshold (float): If the relative velocity of two contact points is bigger than this value, then the dynamic coefficient of friction is applied. Otherwise, the coefficient of friction is interpolated between the static and dynamic one proportional to the relative velocity.
        """

    @overload
    def setMaterialPairProp(self, material1: str, material2: str, friction: float, restitution: float, restitution_threshold: float) -> None:
        """
        Set the material pair properties (friction and restitution).

        Args:
            material1 (str): first material.
            material2 (str): second material.
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            restitution_threshold (float): restitution threshold.
        """

    @overload
    def getMaterialPairProp(self, material1: str, material2: str) -> MaterialPairProperties:
        """
        Get the material pair properties (friction and restitution).

        Args:
            material1 (str): first material.
            material2 (str): second material.

        Returns:
            MaterialPairProperties: material pair properties (friction, restitution, and restitution threshold).
        """

    @overload
    def getMaterialPairProp(self, material1: int, material2: int) -> MaterialPairProperties: ...

    def setDefaultMaterialProperties(self, friction: float, restitution: float, restitution_threshold: float, static_friction: float, static_friction_velocity_threshold: float, rolling_friction: float = 0.0, spinning_friction: float = 0.0) -> None:
        """
        Set the default material properties.

        Args:
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            threshold (float): restitution threshold.
            static friction (float): coefficient of static friction.
            static friction velocity threshold (float): If the relative velocity of two contact points is bigger than this value, then the dynamic coefficient of friction is applied. Otherwise, the coefficient of friction is interpolated between the static and dynamic one proportional to the relative velocity.
            rolling friction (float): coefficient of rolling friction. Defaults to 0.0.
            spinning friction (float): coefficient of spinning friction. Defaults to 0.0.
        """

    def getMaterialIdOrDefault(self, material_name: str) -> int: ...

    def getDefaultMaterialId(self) -> int: ...

class FluidMedium:
    """A fluid: water, air, oil. A density of zero is no fluid."""

    def __init__(self) -> None: ...

    @property
    def density(self) -> float:
        """Density [kg/m^3]."""

    @density.setter
    def density(self, arg: float, /) -> None: ...

    @property
    def viscosity(self) -> float:
        """Dynamic viscosity [Pa s]."""

    @viscosity.setter
    def viscosity(self, arg: float, /) -> None: ...

    @property
    def velocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Velocity of the fluid (a current or the wind) [m/s], in the world frame.
        """

    @velocity.setter
    def velocity(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @staticmethod
    def water() -> FluidMedium:
        """Fresh water at 20 C."""

    @staticmethod
    def seawater() -> FluidMedium:
        """Sea water at 15 C."""

    @staticmethod
    def air() -> FluidMedium:
        """Air at 20 C and sea level."""

class FluidSample:
    """The liquid of a FluidVolume where a body is."""

    def __init__(self) -> None: ...

    @property
    def surfaceHeight(self) -> float:
        """Height of the free surface over the point [m]."""

    @surfaceHeight.setter
    def surfaceHeight(self, arg: float, /) -> None: ...

    @property
    def surfaceNormal(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Normal of the free surface there, in the world frame."""

    @surfaceNormal.setter
    def surfaceNormal(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def velocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Velocity of the liquid at the point [m/s], in the world frame."""

    @velocity.setter
    def velocity(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def acceleration(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Acceleration of the liquid at the point [m/s^2], in the world frame, as in waves.
        """

    @acceleration.setter
    def acceleration(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

class FluidWave:
    """A linear wave on the surface of a deep liquid."""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, direction: Annotated[NDArray[numpy.float64], dict(order='C')], wavelength: float, amplitude: float, phase: float = 0.0) -> None: ...

    @property
    def direction(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Direction the wave travels in, (x, y)."""

    @direction.setter
    def direction(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def wavelength(self) -> float:
        """Distance from crest to crest [m]."""

    @wavelength.setter
    def wavelength(self, arg: float, /) -> None: ...

    @property
    def amplitude(self) -> float:
        """Height of a crest above the mean surface [m]."""

    @amplitude.setter
    def amplitude(self, arg: float, /) -> None: ...

    @property
    def phase(self) -> float:
        """Phase at the origin at time zero [rad]."""

    @phase.setter
    def phase(self, arg: float, /) -> None: ...

class FluidVolume:
    """A body of liquid under a free surface: a pool, a lake, the sea."""

    def __init__(self) -> None: ...

    @property
    def medium(self) -> FluidMedium:
        """The liquid."""

    @medium.setter
    def medium(self, arg: FluidMedium, /) -> None: ...

    @property
    def surfaceHeight(self) -> float:
        """Height of the free surface [m]."""

    @surfaceHeight.setter
    def surfaceHeight(self, arg: float, /) -> None: ...

    @property
    def bottomHeight(self) -> float:
        """Height of the bottom [m]."""

    @bottomHeight.setter
    def bottomHeight(self, arg: float, /) -> None: ...

    @property
    def min(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Lower horizontal corner (x, y) [m]."""

    @min.setter
    def min(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def max(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Upper horizontal corner (x, y) [m]."""

    @max.setter
    def max(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def waveRadiation(self) -> float:
        """
        Scale of the damping of the waves a floating body makes as it bobs; 0 turns it off.
        """

    @waveRadiation.setter
    def waveRadiation(self, arg: float, /) -> None: ...

    @property
    def waves(self) -> list[FluidWave]:
        """Waves on the surface (list of FluidWave); a sampler takes their place."""

    @waves.setter
    def waves(self, arg: Sequence[FluidWave], /) -> None: ...

    def setSampler(self, sampler: Callable[[Annotated[NDArray[numpy.float64], dict(order='C')]], FluidSample]) -> None:
        """
        Give the volume waves and currents: callable(point: np.array[3]) -> FluidSample, the surface and the velocity of the liquid around a point.
        """

    def clearSampler(self) -> None:
        """Remove the sampler."""

    def hasSampler(self) -> bool:
        """Whether the volume has a sampler."""

class FluidCoefficients:
    """How a collision shape interacts with fluids."""

    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool:
        """Whether fluids act on the shape at all."""

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def buoyancy(self) -> bool:
        """Whether the fluid holds the shape up."""

    @buoyancy.setter
    def buoyancy(self, arg: bool, /) -> None: ...

    @property
    def drag(self) -> float:
        """Pressure drag coefficient; negative takes the shape's own."""

    @drag.setter
    def drag(self, arg: float, /) -> None: ...

    @property
    def skinFriction(self) -> float:
        """Skin friction coefficient."""

    @skinFriction.setter
    def skinFriction(self, arg: float, /) -> None: ...

    @property
    def magnusLift(self) -> float:
        """Magnus lift coefficient of a spinning shape."""

    @magnusLift.setter
    def magnusLift(self, arg: float, /) -> None: ...

    @property
    def addedMass(self) -> float:
        """Scale of the added mass."""

    @addedMass.setter
    def addedMass(self, arg: float, /) -> None: ...

class ExplosionProperties:
    """What an explosion is and does."""

    def __init__(self) -> None: ...

    @property
    def chargeMass(self) -> float:
        """TNT-equivalent mass of the charge (kg)."""

    @chargeMass.setter
    def chargeMass(self, arg: float, /) -> None: ...

    @property
    def surfaceBurst(self) -> bool:
        """
        The charge lies on the ground: it loads objects like a free-air charge 1.8 times heavier.
        """

    @surfaceBurst.setter
    def surfaceBurst(self, arg: bool, /) -> None: ...

    @property
    def collisionMask(self) -> int:
        """Collision groups the blast loads."""

    @collisionMask.setter
    def collisionMask(self, arg: int, /) -> None: ...

    @property
    def lineOfSight(self) -> bool:
        """
        Bodies hidden from the charge take only the wave that wraps around them.
        """

    @lineOfSight.setter
    def lineOfSight(self, arg: bool, /) -> None: ...

    @property
    def craterRadius(self) -> float:
        """
        Radius (m) of the crater dug in each heightmap under the charge; zero digs none.
        """

    @craterRadius.setter
    def craterRadius(self, arg: float, /) -> None: ...

    @property
    def craterDepth(self) -> float:
        """Depth (m) of the crater at its center."""

    @craterDepth.setter
    def craterDepth(self, arg: float, /) -> None: ...

class Explosion:
    """An explosion the world has set off."""

    def __init__(self) -> None: ...

    @property
    def id(self) -> int:
        """Increasing identifier, unique within the world."""

    @id.setter
    def id(self, arg: int, /) -> None: ...

    @property
    def time(self) -> float:
        """World time of the detonation (s)."""

    @time.setter
    def time(self, arg: float, /) -> None: ...

    @property
    def properties(self) -> ExplosionProperties:
        """The properties it was set off with."""

    @properties.setter
    def properties(self, arg: ExplosionProperties, /) -> None: ...

    @property
    def projectileId(self) -> int:
        """Identifier of the projectile that set it off, or -1 as size_t."""

    @projectileId.setter
    def projectileId(self, arg: int, /) -> None: ...

    @property
    def position(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Position of the charge in the world frame (m)."""

    @position.setter
    def position(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

class ProjectileProperties:
    """
    A ballistic projectile: a shell, a bullet, a rocket after burnout.

    It flies through the world's fluids: drag acts on its velocity relative to the ambient fluid (whose velocity is the wind)
    or, below the mean surface of a volume of liquid, to the liquid. A world without an ambient fluid is a vacuum. An
    explosive projectile (explosion.chargeMass > 0) detonates where it hits; an inert one stops there. Either hands a
    dynamic body it hits the momentum of an inelastic impact.
    """

    def __init__(self) -> None: ...

    @property
    def mass(self) -> float:
        """Mass (kg)."""

    @mass.setter
    def mass(self, arg: float, /) -> None: ...

    @property
    def radius(self) -> float:
        """Radius (m) of its cross section, which sets its drag area."""

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    @property
    def dragCoefficient(self) -> float:
        """Drag coefficient of its cross section."""

    @dragCoefficient.setter
    def dragCoefficient(self, arg: float, /) -> None: ...

    @property
    def maxLifetime(self) -> float:
        """Seconds after launch at which it is removed."""

    @maxLifetime.setter
    def maxLifetime(self, arg: float, /) -> None: ...

    @property
    def fuseTime(self) -> float:
        """
        Seconds after launch at which it detonates in flight; zero or negative: no time fuse.
        """

    @fuseTime.setter
    def fuseTime(self, arg: float, /) -> None: ...

    @property
    def detonateOnImpact(self) -> bool:
        """It detonates where it hits something, or else stops there."""

    @detonateOnImpact.setter
    def detonateOnImpact(self, arg: bool, /) -> None: ...

    @property
    def explosion(self) -> ExplosionProperties:
        """The explosion it sets off; a chargeMass of zero is inert."""

    @explosion.setter
    def explosion(self, arg: ExplosionProperties, /) -> None: ...

    @property
    def collisionMask(self) -> int:
        """Collision groups the projectile hits."""

    @collisionMask.setter
    def collisionMask(self, arg: int, /) -> None: ...

    @property
    def ignoredObject(self) -> int:
        """World index of an object it passes through."""

    @ignoredObject.setter
    def ignoredObject(self, arg: int, /) -> None: ...

class Projectile:
    """A projectile in flight."""

    def __init__(self) -> None: ...

    @property
    def id(self) -> int:
        """Increasing identifier, unique within the world."""

    @id.setter
    def id(self, arg: int, /) -> None: ...

    @property
    def launchTime(self) -> float:
        """World time of the launch (s)."""

    @launchTime.setter
    def launchTime(self, arg: float, /) -> None: ...

    @property
    def properties(self) -> ProjectileProperties:
        """The properties it was launched with."""

    @properties.setter
    def properties(self, arg: ProjectileProperties, /) -> None: ...

    @property
    def position(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Position in the world frame (m)."""

    @position.setter
    def position(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def velocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Velocity in the world frame (m/s)."""

    @velocity.setter
    def velocity(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def launchPosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Position at the launch (m)."""

    @launchPosition.setter
    def launchPosition(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def launchVelocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Velocity at the launch (m/s)."""

    @launchVelocity.setter
    def launchVelocity(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

class ProjectileHit:
    """A projectile striking an object."""

    def __init__(self) -> None: ...

    @property
    def id(self) -> int:
        """Increasing identifier, unique within the world."""

    @id.setter
    def id(self, arg: int, /) -> None: ...

    @property
    def projectileId(self) -> int:
        """Identifier of the projectile."""

    @projectileId.setter
    def projectileId(self, arg: int, /) -> None: ...

    @property
    def objectIndex(self) -> int:
        """World index of the object hit."""

    @objectIndex.setter
    def objectIndex(self, arg: int, /) -> None: ...

    @property
    def localIdx(self) -> int:
        """Local body index of the body hit."""

    @localIdx.setter
    def localIdx(self, arg: int, /) -> None: ...

    @property
    def mass(self) -> float:
        """Mass of the projectile (kg)."""

    @mass.setter
    def mass(self, arg: float, /) -> None: ...

    @property
    def time(self) -> float:
        """World time of the impact (s)."""

    @time.setter
    def time(self, arg: float, /) -> None: ...

    @property
    def detonated(self) -> bool:
        """The projectile detonated there."""

    @detonated.setter
    def detonated(self, arg: bool, /) -> None: ...

    @property
    def position(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Point of impact in the world frame (m)."""

    @position.setter
    def position(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def normal(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Surface normal at the point of impact."""

    @normal.setter
    def normal(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def velocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """Velocity of the projectile at impact (m/s)."""

    @velocity.setter
    def velocity(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

class SensorType(enum.IntEnum):
    UNKNOWN = 0

    RGB = 1

    DEPTH = 2

    IMU = 3

    SPINNING_LIDAR = 4

    FORCE = 5

    TACTILE = 6

    SEGMENTATION = 7

class SensorMeasurementSource(enum.IntEnum):
    RAISIM = 0

    MANUAL = 1

class Sensor:
    def getPosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getFramePosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getFrameOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getPosInSensorFrame(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getOriInSensorFrame(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getName(self) -> str: ...

    def getFullName(self) -> str: ...

    def getSensorSetModel(self) -> str: ...

    def getSerialNumber(self) -> str: ...

    def setSerialNumber(self, arg: str, /) -> None: ...

    def getType(self) -> SensorType: ...

    def getUpdateRate(self) -> float: ...

    def getUpdateTimeStamp(self) -> float: ...

    def setUpdateRate(self, arg: float, /) -> None: ...

    def setUpdateTimeStamp(self, arg: float, /) -> None: ...

    def updatePose(self) -> None: ...

    def getMeasurementSource(self) -> SensorMeasurementSource: ...

    def setMeasurementSource(self, arg: SensorMeasurementSource, /) -> None: ...

    def getFrameId(self) -> int: ...

    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

class SensorSet:
    def getSensors(self) -> list[Sensor]: ...

    def getSensor(self, arg: str, /) -> Sensor: ...

    def getModel(self) -> str: ...

    def getSerialNumber(self) -> str: ...

    def getName(self) -> str: ...

class CameraProjectionType(enum.Enum):
    PINHOLE = 0

    FISHEYE_EQUIDISTANT = 1

class CameraIntrinsics:
    def __init__(self) -> None: ...

    @property
    def fx(self) -> float: ...

    @fx.setter
    def fx(self, arg: float, /) -> None: ...

    @property
    def fy(self) -> float: ...

    @fy.setter
    def fy(self, arg: float, /) -> None: ...

    @property
    def cx(self) -> float: ...

    @cx.setter
    def cx(self, arg: float, /) -> None: ...

    @property
    def cy(self) -> float: ...

    @cy.setter
    def cy(self, arg: float, /) -> None: ...

    def hasExplicitFocalLength(self) -> bool: ...

    def hasExplicitPrincipalPoint(self) -> bool: ...

class CameraFisheyeDistortion:
    def __init__(self) -> None: ...

    @property
    def k1(self) -> float: ...

    @k1.setter
    def k1(self, arg: float, /) -> None: ...

    @property
    def k2(self) -> float: ...

    @k2.setter
    def k2(self, arg: float, /) -> None: ...

    @property
    def k3(self) -> float: ...

    @k3.setter
    def k3(self, arg: float, /) -> None: ...

    @property
    def k4(self) -> float: ...

    @k4.setter
    def k4(self, arg: float, /) -> None: ...

class CameraLensModel:
    def __init__(self) -> None: ...

    @property
    def projection(self) -> CameraProjectionType: ...

    @projection.setter
    def projection(self, arg: CameraProjectionType, /) -> None: ...

    @property
    def intrinsics(self) -> CameraIntrinsics: ...

    @intrinsics.setter
    def intrinsics(self, arg: CameraIntrinsics, /) -> None: ...

    @property
    def fisheye(self) -> CameraFisheyeDistortion: ...

    @fisheye.setter
    def fisheye(self, arg: CameraFisheyeDistortion, /) -> None: ...

    def isFisheye(self) -> bool: ...

    def projectionName(self) -> str: ...

    def distortionModel(self) -> str: ...

    def setOpenCvFisheye(self, fx: float, fy: float, cx: float, cy: float, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0) -> None: ...

class RGBCameraNoiseType(enum.IntEnum):
    GAUSSIAN = 0

    UNIFORM = 1

    NO_NOISE = 2

class RGBCameraProperties:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def full_name(self) -> str: ...

    @full_name.setter
    def full_name(self, arg: str, /) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def xOffset(self) -> int: ...

    @xOffset.setter
    def xOffset(self, arg: int, /) -> None: ...

    @property
    def yOffset(self) -> int: ...

    @yOffset.setter
    def yOffset(self, arg: int, /) -> None: ...

    @property
    def clipNear(self) -> float: ...

    @clipNear.setter
    def clipNear(self, arg: float, /) -> None: ...

    @property
    def clipFar(self) -> float: ...

    @clipFar.setter
    def clipFar(self, arg: float, /) -> None: ...

    @property
    def hFOV(self) -> float: ...

    @hFOV.setter
    def hFOV(self, arg: float, /) -> None: ...

    @property
    def lens(self) -> CameraLensModel: ...

    @lens.setter
    def lens(self, arg: CameraLensModel, /) -> None: ...

    @property
    def noiseType(self) -> RGBCameraNoiseType: ...

    @noiseType.setter
    def noiseType(self, arg: RGBCameraNoiseType, /) -> None: ...

    @property
    def mean(self) -> float: ...

    @mean.setter
    def mean(self, arg: float, /) -> None: ...

    @property
    def std(self) -> float: ...

    @std.setter
    def std(self, arg: float, /) -> None: ...

    @property
    def shotNoise(self) -> float: ...

    @shotNoise.setter
    def shotNoise(self, arg: float, /) -> None: ...

    @property
    def rowNoiseStd(self) -> float: ...

    @rowNoiseStd.setter
    def rowNoiseStd(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def format(self) -> str: ...

    @format.setter
    def format(self, arg: str, /) -> None: ...

    @staticmethod
    def stringToNoiseType(arg: str, /) -> RGBCameraNoiseType: ...

class RGBCamera(Sensor):
    @staticmethod
    def getStaticType() -> SensorType: ...

    def applyNoise(self) -> None: ...

    def resetNoise(self) -> None: ...

    def setProperties(self, arg: RGBCameraProperties, /) -> None: ...

    def getProperties(self) -> RGBCameraProperties: ...

    def getImageBuffer(self) -> bytes: ...

    @overload
    def setImageBuffer(self, arg: Sequence[str], /) -> None: ...

    @overload
    def setImageBuffer(self, arg: bytes, /) -> None: ...

class DepthCameraNoiseType(enum.IntEnum):
    GAUSSIAN = 0

    UNIFORM = 1

    NO_NOISE = 2

class DepthCameraFrame(enum.IntEnum):
    SENSOR_FRAME = 0

    ROOT_FRAME = 1

    WORLD_FRAME = 2

class DepthCameraProperties:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def full_name(self) -> str: ...

    @full_name.setter
    def full_name(self, arg: str, /) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def xOffset(self) -> int: ...

    @xOffset.setter
    def xOffset(self, arg: int, /) -> None: ...

    @property
    def yOffset(self) -> int: ...

    @yOffset.setter
    def yOffset(self, arg: int, /) -> None: ...

    @property
    def clipNear(self) -> float: ...

    @clipNear.setter
    def clipNear(self, arg: float, /) -> None: ...

    @property
    def clipFar(self) -> float: ...

    @clipFar.setter
    def clipFar(self, arg: float, /) -> None: ...

    @property
    def hFOV(self) -> float: ...

    @hFOV.setter
    def hFOV(self, arg: float, /) -> None: ...

    @property
    def lens(self) -> CameraLensModel: ...

    @lens.setter
    def lens(self, arg: CameraLensModel, /) -> None: ...

    @property
    def noiseType(self) -> DepthCameraNoiseType: ...

    @noiseType.setter
    def noiseType(self, arg: DepthCameraNoiseType, /) -> None: ...

    @property
    def mean(self) -> float: ...

    @mean.setter
    def mean(self, arg: float, /) -> None: ...

    @property
    def std(self) -> float: ...

    @std.setter
    def std(self, arg: float, /) -> None: ...

    @property
    def stdDepthSquared(self) -> float: ...

    @stdDepthSquared.setter
    def stdDepthSquared(self, arg: float, /) -> None: ...

    @property
    def resolution(self) -> float: ...

    @resolution.setter
    def resolution(self, arg: float, /) -> None: ...

    @property
    def dropoutProbability(self) -> float: ...

    @dropoutProbability.setter
    def dropoutProbability(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def format(self) -> str: ...

    @format.setter
    def format(self, arg: str, /) -> None: ...

    @staticmethod
    def stringToNoiseType(arg: str, /) -> DepthCameraNoiseType: ...

class DepthCamera(Sensor):
    @staticmethod
    def getStaticType() -> SensorType: ...

    def applyNoise(self) -> None: ...

    def resetNoise(self) -> None: ...

    def updateRayDirections(self) -> None: ...

    def getDepthArray(self) -> list[float]: ...

    def setDepthArray(self, arg: Sequence[float], /) -> None: ...

    def get3DPoints(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def set3DPoints(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    def getProperties(self) -> DepthCameraProperties: ...

    def depthToPointCloud(self, depthArray: Sequence[float], isInSensorFrame: bool = False) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getPrecomputedRayDir(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

class ImuNoiseType(enum.IntEnum):
    GAUSSIAN = 0

    UNIFORM = 1

    NO_NOISE = 2

class ImuChannelProperties:
    def __init__(self) -> None: ...

    @property
    def bias(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    @bias.setter
    def bias(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> None: ...

    @property
    def noiseDensity(self) -> float: ...

    @noiseDensity.setter
    def noiseDensity(self, arg: float, /) -> None: ...

    @property
    def biasRandomWalk(self) -> float: ...

    @biasRandomWalk.setter
    def biasRandomWalk(self, arg: float, /) -> None: ...

    @property
    def cutoffFrequency(self) -> float: ...

    @cutoffFrequency.setter
    def cutoffFrequency(self, arg: float, /) -> None: ...

    @property
    def resolution(self) -> float: ...

    @resolution.setter
    def resolution(self, arg: float, /) -> None: ...

class ImuProperties:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def full_name(self) -> str: ...

    @full_name.setter
    def full_name(self, arg: str, /) -> None: ...

    @property
    def maxAcc(self) -> float: ...

    @maxAcc.setter
    def maxAcc(self, arg: float, /) -> None: ...

    @property
    def maxAngVel(self) -> float: ...

    @maxAngVel.setter
    def maxAngVel(self, arg: float, /) -> None: ...

    @property
    def noiseType(self) -> ImuNoiseType: ...

    @noiseType.setter
    def noiseType(self, arg: ImuNoiseType, /) -> None: ...

    @property
    def mean(self) -> float: ...

    @mean.setter
    def mean(self, arg: float, /) -> None: ...

    @property
    def std(self) -> float: ...

    @std.setter
    def std(self, arg: float, /) -> None: ...

    @property
    def accelerometer(self) -> ImuChannelProperties: ...

    @accelerometer.setter
    def accelerometer(self, arg: ImuChannelProperties, /) -> None: ...

    @property
    def gyroscope(self) -> ImuChannelProperties: ...

    @gyroscope.setter
    def gyroscope(self, arg: ImuChannelProperties, /) -> None: ...

    @property
    def latency(self) -> float: ...

    @latency.setter
    def latency(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @staticmethod
    def stringToNoiseType(arg: str, /) -> ImuNoiseType: ...

class InertialMeasurementUnit(Sensor):
    @staticmethod
    def getStaticType() -> SensorType: ...

    def setProperties(self, arg: ImuProperties, /) -> None: ...

    def reset(self) -> None: ...

    def resetNoise(self) -> None: ...

    def getDroppedMeasurementCount(self) -> int: ...

    def getDeliveryTimeStamp(self) -> float: ...

    def getGroundTruthOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getAccelerometerBias(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def getGyroscopeBias(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def getLinearAcceleration(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def getAngularVelocity(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def getOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def setLinearAcceleration(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> None: ...

    def setAngularVelocity(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> None: ...

    def setOrientation(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    def getProperties(self) -> ImuProperties: ...

class SpinningLidarProperties:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def full_name(self) -> str: ...

    @full_name.setter
    def full_name(self, arg: str, /) -> None: ...

    @property
    def yawSamples(self) -> int: ...

    @yawSamples.setter
    def yawSamples(self, arg: int, /) -> None: ...

    @property
    def pitchSamples(self) -> int: ...

    @pitchSamples.setter
    def pitchSamples(self, arg: int, /) -> None: ...

    @property
    def pitchMinAngle(self) -> float: ...

    @pitchMinAngle.setter
    def pitchMinAngle(self, arg: float, /) -> None: ...

    @property
    def pitchMaxAngle(self) -> float: ...

    @pitchMaxAngle.setter
    def pitchMaxAngle(self, arg: float, /) -> None: ...

    @property
    def rangeMin(self) -> float: ...

    @rangeMin.setter
    def rangeMin(self, arg: float, /) -> None: ...

    @property
    def rangeMax(self) -> float: ...

    @rangeMax.setter
    def rangeMax(self, arg: float, /) -> None: ...

    @property
    def spinningRate(self) -> float: ...

    @spinningRate.setter
    def spinningRate(self, arg: float, /) -> None: ...

    @property
    def spinDirection(self) -> int: ...

    @spinDirection.setter
    def spinDirection(self, arg: int, /) -> None: ...

    @property
    def rangeBias(self) -> float: ...

    @rangeBias.setter
    def rangeBias(self, arg: float, /) -> None: ...

    @property
    def rangeNoiseStd(self) -> float: ...

    @rangeNoiseStd.setter
    def rangeNoiseStd(self, arg: float, /) -> None: ...

    @property
    def angularNoiseStd(self) -> float: ...

    @angularNoiseStd.setter
    def angularNoiseStd(self, arg: float, /) -> None: ...

    @property
    def resolution(self) -> float: ...

    @resolution.setter
    def resolution(self, arg: float, /) -> None: ...

    @property
    def dropoutProbability(self) -> float: ...

    @dropoutProbability.setter
    def dropoutProbability(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

class SpinningLidar(Sensor):
    @staticmethod
    def getStaticType() -> SensorType: ...

    def applyNoise(self) -> None: ...

    def resetNoise(self) -> None: ...

    def getProperties(self) -> SpinningLidarProperties: ...

    def setProperties(self, arg: SpinningLidarProperties, /) -> None: ...

    def getScan(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def setScan(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    def getTimeStamp(self) -> float: ...

    def setTimeStamp(self, arg: float, /) -> None: ...

    def getCurrentYaw(self) -> float: ...

class WrenchSensorModel:
    def __init__(self) -> None: ...

    @property
    def forceBias(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    @forceBias.setter
    def forceBias(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> None: ...

    @property
    def torqueBias(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    @torqueBias.setter
    def torqueBias(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], /) -> None: ...

    @property
    def forceNoiseStd(self) -> float: ...

    @forceNoiseStd.setter
    def forceNoiseStd(self, arg: float, /) -> None: ...

    @property
    def torqueNoiseStd(self) -> float: ...

    @torqueNoiseStd.setter
    def torqueNoiseStd(self, arg: float, /) -> None: ...

    @property
    def maxForce(self) -> float: ...

    @maxForce.setter
    def maxForce(self, arg: float, /) -> None: ...

    @property
    def maxTorque(self) -> float: ...

    @maxTorque.setter
    def maxTorque(self, arg: float, /) -> None: ...

    @property
    def cutoffFrequency(self) -> float: ...

    @cutoffFrequency.setter
    def cutoffFrequency(self, arg: float, /) -> None: ...

    @property
    def latency(self) -> float: ...

    @latency.setter
    def latency(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

class WrenchSensor(Sensor):
    def getForce(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def getTorque(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def getAcquisitionStartTime(self) -> float: ...

    def getDeliveryTimeStamp(self) -> float: ...

    def getDroppedMeasurementCount(self) -> int: ...

    def reset(self) -> None: ...

    def resetNoise(self) -> None: ...

    def update(self, arg: World, /) -> None: ...

class ForceSensorProperties:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def full_name(self) -> str: ...

    @full_name.setter
    def full_name(self, arg: str, /) -> None: ...

    @property
    def joint(self) -> str: ...

    @joint.setter
    def joint(self, arg: str, /) -> None: ...

    @property
    def model(self) -> WrenchSensorModel: ...

    @model.setter
    def model(self, arg: WrenchSensorModel, /) -> None: ...

class ForceSensor(WrenchSensor):
    @staticmethod
    def getStaticType() -> SensorType: ...

    def getProperties(self) -> ForceSensorProperties: ...

    def setProperties(self, arg: ForceSensorProperties, /) -> None: ...

class TactileSensorProperties:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def full_name(self) -> str: ...

    @full_name.setter
    def full_name(self, arg: str, /) -> None: ...

    @property
    def link(self) -> str: ...

    @link.setter
    def link(self, arg: str, /) -> None: ...

    @property
    def collision(self) -> str: ...

    @collision.setter
    def collision(self, arg: str, /) -> None: ...

    @property
    def width(self) -> float: ...

    @width.setter
    def width(self, arg: float, /) -> None: ...

    @property
    def height(self) -> float: ...

    @height.setter
    def height(self, arg: float, /) -> None: ...

    @property
    def depth(self) -> float: ...

    @depth.setter
    def depth(self, arg: float, /) -> None: ...

    @property
    def columns(self) -> int: ...

    @columns.setter
    def columns(self, arg: int, /) -> None: ...

    @property
    def rows(self) -> int: ...

    @rows.setter
    def rows(self, arg: int, /) -> None: ...

    @property
    def model(self) -> WrenchSensorModel: ...

    @model.setter
    def model(self, arg: WrenchSensorModel, /) -> None: ...

class TactileSensor(WrenchSensor):
    @staticmethod
    def getStaticType() -> SensorType: ...

    def getProperties(self) -> TactileSensorProperties: ...

    def setProperties(self, arg: TactileSensorProperties, /) -> None: ...

    def getNormalForce(self) -> float: ...

    def hasCenterOfPressure(self) -> bool: ...

    def getCenterOfPressure(self) -> Annotated[NDArray[numpy.float64], dict(shape=(2), order='C')]: ...

    def getTaxelForces(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getTaxelSlipVelocities(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

class SegmentationType(enum.IntEnum):
    SEMANTIC = 0

    INSTANCE = 1

    PANOPTIC = 2

class SegmentationCameraProperties:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def full_name(self) -> str: ...

    @full_name.setter
    def full_name(self, arg: str, /) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def xOffset(self) -> int: ...

    @xOffset.setter
    def xOffset(self, arg: int, /) -> None: ...

    @property
    def yOffset(self) -> int: ...

    @yOffset.setter
    def yOffset(self, arg: int, /) -> None: ...

    @property
    def clipNear(self) -> float: ...

    @clipNear.setter
    def clipNear(self, arg: float, /) -> None: ...

    @property
    def clipFar(self) -> float: ...

    @clipFar.setter
    def clipFar(self, arg: float, /) -> None: ...

    @property
    def hFOV(self) -> float: ...

    @hFOV.setter
    def hFOV(self, arg: float, /) -> None: ...

    @property
    def lens(self) -> CameraLensModel: ...

    @lens.setter
    def lens(self, arg: CameraLensModel, /) -> None: ...

    @property
    def segmentationType(self) -> SegmentationType: ...

    @segmentationType.setter
    def segmentationType(self, arg: SegmentationType, /) -> None: ...

    @staticmethod
    def stringToSegmentationType(arg: str, /) -> SegmentationType: ...

class SegmentationCamera(Sensor):
    @staticmethod
    def getStaticType() -> SensorType: ...

    def getProperties(self) -> SegmentationCameraProperties: ...

    def setProperties(self, properties: SegmentationCameraProperties) -> None: ...

    def update(self, world: World) -> None: ...

    def getObjectImage(self) -> Annotated[NDArray[numpy.int32], dict(order='C')] | None:
        """
        World index of the object in each pixel as a (height, width) int32 array; None for SEMANTIC.
        """

    def getBodyImage(self) -> Annotated[NDArray[numpy.int32], dict(order='C')] | None:
        """
        Body index within the object in each pixel as a (height, width) int32 array; None for SEMANTIC.
        """

    def getLabelImage(self) -> Annotated[NDArray[numpy.int32], dict(order='C')] | None:
        """
        Segmentation label in each pixel as a (height, width) int32 array; None for INSTANCE.
        """

    def setObjectImage(self, image: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)]) -> None: ...

    def setBodyImage(self, image: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)]) -> None: ...

    def setLabelImage(self, image: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)]) -> None: ...

    @staticmethod
    def hasInstanceImages(type: SegmentationType) -> bool: ...

    @staticmethod
    def hasLabelImage(type: SegmentationType) -> bool: ...

    @staticmethod
    def imageCount(type: SegmentationType) -> int: ...

    @staticmethod
    def colorOf(id: int) -> list[int]: ...

    @staticmethod
    def colorize(ids: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], background: Sequence[int] = [0, 0, 0, 0]) -> Annotated[NDArray[numpy.uint8], dict(order='C')]:
        """
        RGBA uint8 colors of an id image (shape + (4,)); negative ids get background.
        """

    BACKGROUND: int = -1

    NOT_WORLD_OBJECT: int = -2

class ObjectType(enum.IntEnum):
    SPHERE = 0

    BOX = 1

    CYLINDER = 2

    CONE = 3

    CAPSULE = 4

    MESH = 5

    HALFSPACE = 6

    COMPOUND = 7

    HEIGHTMAP = 8

    ARTICULATED_SYSTEM = 9

    DEFORMABLE = 10

    GRANULAR = 11

    UNRECOGNIZED = 12

class MeshCollisionMode(enum.IntEnum):
    ORIGINAL_MESH = 0

    CONVEX_HULL = 1

    CONVEXIFY = 2

class BodyType(enum.IntEnum):
    STATIC = 1

    KINEMATIC = 2

    DYNAMIC = 3

class Object:
    """Raisim Object from which all other objects/bodies inherit from."""

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    def getName(self) -> str:
        """
        Get the object's name.

        Returns:
            str: object's name.
        """

    def setName(self, name: str) -> None:
        """
        Set the object's name.

        Args:
            name (str): object's name.
        """

    def setIndexInWorld(self, index: int) -> None:
        """
        Set the object index in the world.

        Args:
            index (int): object index.
        """

    def getIndexInWorld(self) -> int:
        """
        Get the object index in the world.

        Returns:
            int: object index in the world.
        """

    def setSegmentationLabel(self, label: int) -> None:
        """
        Set the semantic label segmentation cameras report for this object (0 means unlabeled).

        Args:
            label (int): non-negative label, e.g. from World.addSegmentationClass().
        """

    def getSegmentationLabel(self) -> int:
        """
        Get the semantic label set by setSegmentationLabel().

        Returns:
            int: the label (0 if none).
        """

    def getId(self) -> int: ...

    def getFluidForce(self, local_idx: int = 0) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the force the fluids exerted on a body in the last step, in the world frame.

        Args:
            local_idx (int): body index (0 for single bodies).

        Returns:
            np.array[float[3]]: force [N].
        """

    def getFluidTorque(self, local_idx: int = 0) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the torque the fluids exerted on a body in the last step about its center of mass, in the world frame.

        Args:
            local_idx (int): body index (0 for single bodies).

        Returns:
            np.array[float[3]]: torque [N m].
        """

    @overload
    def getContacts(self) -> list[contact.Contact]:
        """
        Get the list of contact points.

        Returns:
            list[Contact]: list of contact points.
        """

    @overload
    def getContacts(self) -> list[contact.Contact]: ...

    @overload
    def setExternalForce(self, local_idx: int, force: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the external force on the body.

        Args:
            local_idx (int): local index.
            force (np.array[float[3]]): force vector.
        """

    @overload
    def setExternalForce(self, local_idx: int, position: Annotated[NDArray[numpy.float64], dict(order='C')], force: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the external force at a position on the body.

        Args:
            local_idx (int): local index.
            position (np.array[float[3]]): position in the body frame.
            force (np.array[float[3]]): force vector in world frame.
        """

    def setExternalTorque(self, local_idx: int, torque: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the external force on the body.

        Args:
            local_idx (int): local index.
            force (np.array[float[3]]): force vector.
        """

    def setConstraintForce(self, local_idx: int, position: Annotated[NDArray[numpy.float64], dict(order='C')], force: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the constraint force at a position on the body.

        Args:
            local_idx (int): local index.
            position (np.array[float[3]]): position in the body frame.
            force (np.array[float[3]]): force vector in world frame.
        """

    def getMass(self, local_idx: int) -> float: ...

    def getObjectType(self) -> ObjectType: ...

    @overload
    def getPosition(self, local_idx: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the world position.

        Args:
            local_idx (int): local index.

        Returns:
            np.array[float[3]]: position expressed in the Cartesian world frame.
        """

    @overload
    def getPosition(self, arg0: int, arg1: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @overload
    def getVelocity(self, local_idx: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the world linear velocity.

        Args:
            local_idx (int): local index.

        Returns:
            np.array[float[3]]: linear velocity expressed in the Cartesian world frame.
        """

    @overload
    def getVelocity(self, local_idx: int, body_pos: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getOrientation(self, local_idx: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the world orientation as a rotation matrix.

        Args:
            local_idx (int): local index.

        Returns:
            np.array[float[3,3]]: rotation matrix.
        """

    @overload
    def getBodyType(self, arg: int, /) -> BodyType: ...

    @overload
    def getBodyType(self) -> BodyType: ...

    def getContactPointVel(self, point_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the contact point velocity.

        Args:
            point_id (int): point id.

        Returns:
            np.array[float[3]]: contact point velocity.
        """

    def getContactAabbCount(self) -> int: ...

    def getContactAabb(self, idx: int) -> contact.AABB: ...

    def getExternalForce(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getExternalForcePosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getExternalTorque(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getExternalTorquePosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def clearExternalForcesAndTorques(self) -> None: ...

    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

class GyroscopicMode(enum.IntEnum):
    IMPLICIT_GYROSCOPIC_FORCE_BODY = 0

    IMPLICIT_GYROSCOPIC_FORCE_WORLD = 1

    EXPLICIT_GYROSCOPIC_FORCE = 2

    NO_GYROSCOPIC_FORCE = 3

class SingleBodyObject(Object):
    """
    Raisim Single Object from which all single objects/bodies (such as box, sphere, etc) inherit from.
    """

    def getQuaternion(self) -> Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')]:
        """
        Get the body's orientation (expressed as a quaternion [w,x,y,z]) with respect to the world frame.

        Returns:
            np.array[float[4]]: quaternion [w,x,y,z].
        """

    def getRotationMatrix(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3, 3), order='F')]:
        """
        Get the body's orientation (expressed as a rotation matrix) with respect to the world frame.

        Returns:
            np.array[float[3,3]]: rotation matrix.
        """

    def getOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getPosition(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]:
        """
        Get the body's position with respect to the world frame.

        Returns:
            np.array[float[3]]: position in the world frame.
        """

    def getComPosition(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]:
        """
        Get the body's center of mass position with respect to the world frame.

        Returns:
            np.array[float[3]]: center of mass position in the world frame.
        """

    def getComPosition_rs(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getBodyToComPosition_rs(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getLinearVelocity(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]:
        """
        Get the body's linear velocity with respect to the world frame.

        Returns:
            np.array[float[3]]: linear velocity in the world frame.
        """

    def getAngularVelocity(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]:
        """
        Get the body's angular velocity position with respect to the world frame.

        Returns:
            np.array[float[3]]: angular velocity in the world frame.
        """

    def getKineticEnergy(self) -> float:
        """
        Get the body's kinetic energy.

        Returns:
            float: kinetic energy.
        """

    def getPotentialEnergy(self, gravity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> float:
        """
        Get the body's potential energy due to gravity.

        Args:
            gravity (np.array[float[3]]): gravity vector.

        Returns:
            float: potential energy.
        """

    def getEnergy(self, gravity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> float:
        """
        Get the body's total energy.

        Args:
            gravity (np.array[float[3]]): gravity vector.

        Returns:
            float: total energy.
        """

    def getLinearMomentum(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]:
        """
        Get the body's linear momentum.

        Returns:
            np.array[float[3]]: linear momentum.
        """

    def getMass(self, local_idx: int) -> float:
        """
        Get the body's mass.

        Args:
            local_idx (int): local index.

        Returns:
            float: mass (kg).
        """

    def setMass(self, mass: float) -> None: ...

    def getInertiaMatrix_W(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3, 3), order='F')]:
        """
        Get the body's inertia matrix expressed in the world frame.

        Returns:
            np.array[float[3,3]]: world inertia matrix.
        """

    def getInertiaMatrix_B(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3, 3), order='F')]:
        """
        Get the body's inertia matrix expressed in the body frame.

        Returns:
            np.array[float[3,3]]: body inertia matrix.
        """

    def setInertia(self, inertia: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def getCom(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def setCom(self, com: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def getObjectType(self) -> ObjectType:
        """
        Get the body's type.

        Returns:
            raisim.ObjectType: object type (BOX, CYLINDER, CAPSULE, CONE, SPHERE, etc.)
        """

    def getCollisionObject(self) -> CollisionBodyHandle:
        """
        Get the collision object.

        Returns:
            CollisionBodyHandle: collision body handle.
        """

    def getCollisionBody(self) -> CollisionBodyHandle: ...

    def getGyroscopicMode(self) -> GyroscopicMode:
        """
        Get the gyroscopic mode.

        Returns:
            raisim.GyroscopicMode: gyroscopic mode (between ['IMPLICIT_GYROSCOPIC_FORCE_BODY',
            'IMPLICIT_GYROSCOPIC_FORCE_WORLD', 'EXPLICIT_GYROSCOPIC_FORCE', 'NO_GYROSCOPIC_FORCE'])
        """

    @overload
    def setPosition(self, origin_position: Annotated[NDArray[numpy.float64], dict(shape=(3), writable=False)]) -> None:
        """
        Set the specified origin position.

        Args:
            origin_position (np.array[float[3]]): origin position.
        """

    @overload
    def setPosition(self, x: float, y: float, z: float) -> None:
        """
        Set the specified origin position.

        	    Args:
                x (float): x component of the origin position.
                y (float): y component of the origin position.
                z (float): z component of the origin position.
        """

    @overload
    def setOrientation(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """
        Set the specified orientation (expressed as a quaternion [x,y,z,w]) for the body.

        Args:
            w (float): scalar component of the quaternion.
            x (float): x component of the vector in the quaternion.
            y (float): y component of the vector in the quaternion.
            z (float): z component of the vector in the quaternion.
        """

    @overload
    def setOrientation(self, rotation_matrix: Annotated[NDArray[numpy.float64], dict(shape=(3, 3), order='F')]) -> None:
        """
        Set the specified orientation (expressed as a rotation matrix) for the body.

        Args:
            rotation_matrix (np.array[float[3,3]]): rotation matrix.
        """

    @overload
    def setOrientation(self, quaternion: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the specified orientation (expressed as a quaternion [x,y,z,w]) for the body.

        Args:
            quaternion (np.array[float[4]]): quaternion [x,y,z,w].
        """

    @overload
    def setPose(self, position: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], rotation_matrix: Annotated[NDArray[numpy.float64], dict(shape=(3, 3), order='F')]) -> None:
        """
        Set the specified pose for the body.

        Args:
            position (np.array[float[3]]): origin position vector.
            rotation_matrix (np.array[float[3,3]]): rotation matrix.
        """

    @overload
    def setPose(self, position: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], quaternion: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the specified pose for the body.

        Args:
            position (np.array[float[3]]): origin position vector.
            quaternion (np.array[float[4]]): quaternion (expressed as [w,x,y,z]).
        """

    @overload
    def setVelocity(self, linear_velocity: Annotated[NDArray[numpy.float64], dict(shape=(3), writable=False)], angular_velocity: Annotated[NDArray[numpy.float64], dict(shape=(3), writable=False)]) -> None:
        """
        Set the specified linear and angular velocities for the body.

        Args:
            linear_velocity (np.array[float[3]]): linear velocity.
            angular_velocity (np.array[float[3,3]]): angular velocity.
        """

    @overload
    def setVelocity(self, dx: float, dy: float, dz: float, wx: float, wy: float, wz: float) -> None:
        """
        Set the specified linear and angular velocities for the body.

        Args:
            dx (float): x component of the linear velocity.
            dy (float): y component of the linear velocity.
            dz (float): z component of the linear velocity.
            wx (float): x component of the angular velocity.
            wy (float): y component of the angular velocity.
            wz (float): z component of the angular velocity.
        """

    def setLinearVelocity(self, linear_velocity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def setAngularVelocity(self, angular_velocity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    @overload
    def setExternalForce(self, local_idx: int, force: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    @overload
    def setExternalForce(self, local_idx: int, pos: Annotated[NDArray[numpy.float64], dict(order='C')], force: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the external force on the body.

        Args:
            local_idx (int): local index.
            force (np.array[float[3]]): force vector.
        """

    def setConstraintForce(self, local_idx: int, pos: Annotated[NDArray[numpy.float64], dict(order='C')], force: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def setExternalTorque(self, local_idx: int, torque: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the external force on the body.

        Args:
            local_idx (int): local index.
            force (np.array[float[3]]): force vector.
        """

    def setGyroscopicMode(self, mode: GyroscopicMode) -> None:
        """
        Set the gyroscopic mode for the body.

        Args:
            mode (GyroscopicMode): gyroscopic mode (select between (between [GyroscopicMode.IMPLICIT_GYROSCOPIC_FORCE_BODY,
                GyroscopicMode.IMPLICIT_GYROSCOPIC_FORCE_WORLD, GyroscopicMode.EXPLICIT_GYROSCOPIC_FORCE,
                GyroscopicMode.NO_GYROSCOPIC_FORCE])
        """

    def getContactPointVel(self, point_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the contact point velocity.

        Args:
            point_id (int): point id.

        Returns:
            np.array[float[3]]: contact point velocity.
        """

    def updateCollision(self) -> None:
        """Update the collisions."""

    def setLinearDamping(self, damping: float) -> None:
        """
        Set the body's linear damping coefficient.

        Args:
            damping (float): linear damping coefficient.
        """

    def setAngularDamping(self, damping: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the body's angular damping.

        Args:
            damping (np.array[float[3]]): angular damping.
        """

    def setBodyType(self, body_type: BodyType) -> None:
        """
        Set the body's type.

        Args:
            body_type (BodyType): body type.
        """

    def setAppearance(self, appearance: str) -> None:
        """
        Set the appearance

        Args:
            appearance (string): can be a color name (e.g., brown) or a color code (e.g., "150,200,100,150" which is in an RGBA format).
        """

    def getAppearance(self) -> str: ...

    def getCollisionGroup(self) -> int:
        """
        Get the body's collision group.

        Returns:
            int: collision group.
        """

    def getCollisionMask(self) -> int:
        """
        Get the body's collision mask.

        Returns:
            int: collision mask.
        """

class Box(SingleBodyObject):
    """Raisim Box."""

    def __init__(self, x: float, y: float, z: float, mass: float) -> None:
        """
        Initialize a box.

        Args:
            x (float): length along the x axis.
            y (float): length along the y axis.
            z (float): length along the z axis.
            mass (float): mass of the box.
        """

    def getDim(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the box's dimensions.

        Returns:
            tuple[float[3]]: dimensions along each axis.
        """

class Capsule(SingleBodyObject):
    """Raisim Capsule."""

    def __init__(self, radius: float, height: float, mass: float) -> None:
        """
        Initialize a capsule.

        Args:
            radius (float): radius of the capsule.
            height (float): height of the capsule.
            mass (float): mass of the capsule.
        """

    def getRadius(self) -> float:
        """
        Get the capsule's radius.

        Returns:
            float: radius of the capsule.
        """

    def getHeight(self) -> float:
        """
        Get the capsule's height.

        Returns:
            float: height of the capsule.
        """

class Cone(SingleBodyObject):
    """Raisim Cone."""

    def __init__(self, radius: float, height: float, mass: float) -> None:
        """
        Initialize a cone.

        Args:
            radius (float): radius of the cone.
            height (float): height of the cone.
            mass (float): mass of the cone.
        """

    def getRadius(self) -> float: ...

    def getHeight(self) -> float: ...

class Compound(SingleBodyObject):
    """Raisim Compound bodies."""

    def __init__(self, children: Sequence[Compound.CompoundObjectChild], mass: float, center_of_mass: Annotated[NDArray[numpy.float64], dict(order='C')], inertia: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Initialize the Compound object.

        Args:
            children (list[CompoundObjectChild]): list of child objects.
            mass (float): total mass of the compound object.
            center_of_mass (np.array[float[3]]): center of mass of the compound object.
            inertia (np.array[float[3,3]]): total inertia matrix of the compound object.
        """

    class CompoundObjectChild:
        """Raisim Compound object child."""

        @property
        def childCompound(self) -> str: ...

        @childCompound.setter
        def childCompound(self, arg: str, /) -> None: ...

        @property
        def objectType(self) -> ObjectType: ...

        @objectType.setter
        def objectType(self, arg: ObjectType, /) -> None: ...

        @property
        def objectParam(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @objectParam.setter
        def objectParam(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

        @property
        def material(self) -> str: ...

        @material.setter
        def material(self, arg: str, /) -> None: ...

        @property
        def appearance(self) -> str: ...

        @appearance.setter
        def appearance(self, arg: str, /) -> None: ...

        @property
        def trans(self) -> Transformation: ...

        @trans.setter
        def trans(self, arg: Transformation, /) -> None: ...

    def getObjList(self) -> list[Compound.CompoundObjectChild]: ...

    def getCollisionBodyList(self) -> list[CollisionBodyHandle]: ...

class Cylinder(SingleBodyObject):
    """Raisim Cylinder."""

    def __init__(self, radius: float, height: float, mass: float) -> None:
        """
        Initialize a cylinder.

        Args:
            radius (float): radius of the cylinder.
            height (float): height of the cylinder.
            mass (float): mass of the cylinder.
        """

    def getRadius(self) -> float:
        """
        Get the cylinder's radius.

        Returns:
            float: radius of the cylinder.
        """

    def getHeight(self) -> float:
        """
        Get the cylinder's height.

        Returns:
            float: height of the cylinder.
        """

class Mesh(SingleBodyObject):
    """Raisim Mesh."""

    def __init__(self, filename: str, mass: float, scale: float = 1.0) -> None:
        """
        Initialize a mesh.

        Args:
            filename (str): mesh file path.
            mass (float): mass.
            scale (float): scale factor.
        """

    def getMeshFileName(self) -> str:
        """
        Get the mesh's full path.

        Returns:
            float: full path of the mesh file.
        """

    def getVerticies(self) -> list[float]: ...

    def getIndicies(self) -> list[int]: ...

    def getScale(self) -> float: ...

    @staticmethod
    def loadMesh(filename: str, scale: float = 1.0) -> tuple: ...

    @staticmethod
    def loadObj(filename: str, scale: float = 1.0) -> tuple: ...

class Sphere(SingleBodyObject):
    """Raisim Sphere."""

    def __init__(self, radius: float, mass: float) -> None:
        """
        Initialize a sphere.

        Args:
            radius (float): radius of the sphere.
            mass (float): mass of the sphere.
        """

    def getRadius(self) -> float:
        """
        Get the sphere's radius.

        Returns:
            float: radius of the sphere.
        """

class ControlMode(enum.IntEnum):
    FORCE_AND_TORQUE = 0

    PD_PLUS_FEEDFORWARD_TORQUE = 1

class CollisionBodyHandle:
    """Raisim collision body handle."""

    def __init__(self) -> None: ...

    @property
    def objectType(self) -> ObjectType: ...

    @objectType.setter
    def objectType(self, arg: ObjectType, /) -> None: ...

    @property
    def bodyType(self) -> BodyType: ...

    @bodyType.setter
    def bodyType(self, arg: BodyType, /) -> None: ...

    @property
    def objectIndex(self) -> int: ...

    @objectIndex.setter
    def objectIndex(self, arg: int, /) -> None: ...

    @property
    def localIdx(self) -> int: ...

    @localIdx.setter
    def localIdx(self, arg: int, /) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def material(self) -> str: ...

    @material.setter
    def material(self, arg: str, /) -> None: ...

    @property
    def materialVersion(self) -> int: ...

    @materialVersion.setter
    def materialVersion(self, arg: int, /) -> None: ...

    @property
    def materialId(self) -> int: ...

    @materialId.setter
    def materialId(self, arg: int, /) -> None: ...

class CollisionDefinition:
    """Raisim CollisionDefinition struct."""

    @property
    def rotOffset(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @rotOffset.setter
    def rotOffset(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def posOffset(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @posOffset.setter
    def posOffset(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def localIdx(self) -> int: ...

    @localIdx.setter
    def localIdx(self, arg: int, /) -> None: ...

    @property
    def shape(self) -> ShapeType: ...

    @shape.setter
    def shape(self, arg: ShapeType, /) -> None: ...

    @property
    def shapeParam(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @shapeParam.setter
    def shapeParam(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def scale(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @scale.setter
    def scale(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def meshIndex(self) -> int: ...

    @meshIndex.setter
    def meshIndex(self, arg: int, /) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    def setMaterial(self, material: str) -> None: ...

    def getMaterial(self) -> str: ...

    def setCollisionGroup(self, group: int) -> None: ...

    def setCollisionMask(self, mask: int) -> None: ...

    def getCollisionGroup(self) -> int: ...

    def getCollisionMask(self) -> int: ...

    def getCollisionBody(self) -> CollisionBodyHandle: ...

class ShapeType(enum.IntEnum):
    Box = 0

    Cylinder = 1

    Sphere = 2

    Mesh = 3

    Capsule = 4

    Cone = 5

class JointType(enum.IntEnum):
    FIXED = 0

    REVOLUTE = 1

    PRISMATIC = 2

    SPHERICAL = 3

    FLOATING = 4

class VisObject:
    """Raisim visual object definition."""

    @property
    def visShapeParam(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @visShapeParam.setter
    def visShapeParam(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def offset(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @offset.setter
    def offset(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def rot(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @rot.setter
    def rot(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def color(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @color.setter
    def color(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def scale(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @scale.setter
    def scale(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def shape(self) -> ShapeType: ...

    @shape.setter
    def shape(self, arg: ShapeType, /) -> None: ...

    @property
    def fileName(self) -> str: ...

    @fileName.setter
    def fileName(self, arg: str, /) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def localIdx(self) -> int: ...

    @localIdx.setter
    def localIdx(self, arg: int, /) -> None: ...

    @property
    def material(self) -> str: ...

    @material.setter
    def material(self, arg: str, /) -> None: ...

class SpringElement:
    """Spring element in an articulated system"""

    @property
    def q_ref(self) -> Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')]: ...

    @q_ref.setter
    def q_ref(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')], /) -> None: ...

    @property
    def childBodyId(self) -> int: ...

    @childBodyId.setter
    def childBodyId(self, arg: int, /) -> None: ...

    @property
    def stiffness(self) -> float: ...

    @stiffness.setter
    def stiffness(self, arg: float, /) -> None: ...

class CoordinateFrame:
    """Raisim Coordinate Frame class."""

    @property
    def position(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @position.setter
    def position(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def orientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @orientation.setter
    def orientation(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def parentId(self) -> int: ...

    @parentId.setter
    def parentId(self, arg: int, /) -> None: ...

    @property
    def parentName(self) -> str: ...

    @parentName.setter
    def parentName(self, arg: str, /) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def isChild(self) -> bool: ...

    @isChild.setter
    def isChild(self, arg: bool, /) -> None: ...

class LinkRef:
    def getPosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getPose(self) -> tuple: ...

    def getCollisionDefinition(self, name: str) -> CollisionDefinition: ...

    def getVisualObject(self, name: str) -> VisObject: ...

    def setWeight(self, weight: float) -> None: ...

    def getWeight(self) -> float: ...

    def setInertia(self, inertia: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def getInertia(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def setComPositionInParentFrame(self, com: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def getComPositionInParentFrame(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getCollisionSet(self) -> dict[str, CollisionDefinition]: ...

    def getVisualSet(self) -> dict[str, VisObject]: ...

class JointRef:
    def getPosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getPose(self) -> tuple: ...

    def getJointCoordinate(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getJointAngle(self) -> float: ...

    def getPositionInParentFrame(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getJointAxis(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getType(self) -> JointType: ...

    def getIdxInGeneralizedCoordinate(self) -> int: ...

    def getLinearVelocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

class ArticulatedSystemOption:
    """Raisim Articulated System Option."""

    def __init__(self) -> None: ...

    @property
    def do_not_collide_with_parent(self) -> bool: ...

    @do_not_collide_with_parent.setter
    def do_not_collide_with_parent(self, arg: bool, /) -> None: ...

    @property
    def convexify_collision_meshes(self) -> bool: ...

    @convexify_collision_meshes.setter
    def convexify_collision_meshes(self, arg: bool, /) -> None: ...

class ArticulatedSystem(Object):
    """Raisim Articulated System."""

    @overload
    def __init__(self) -> None:
        """Initialize the Articulated System."""

    @overload
    def __init__(self, filename: str, resource_directory: str, joint_order: Sequence[str], options: ArticulatedSystemOption) -> None:
        """
        Initialize the Articulated System.

        Do not call this method yourself. use World class to create an Articulated system.

        Args:
            filename (str): path to the robot description file (URDF, etc).
            resource_directory (str): path the resource directory. If empty, it will use the robot description folder.
            joint_order (list[str]): specify the joint order, if we want it to be different from the URDF file.
            options (ArticulatedSystemOption): options.
        """

    class Frame(enum.Enum):
        WORLD_FRAME = 0

        PARENT_FRAME = 1

        BODY_FRAME = 2

    class IntegrationScheme(enum.Enum):
        TRAPEZOID = 0

        SEMI_IMPLICIT = 1

        EULER = 2

        RUNGE_KUTTA_4 = 3

    def getGeneralizedCoordinate(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the generalized coordinates of the system.

        The dimension of the returned vector is equal to the number of degrees of freedom that each joint provides.
        A floating joint provides 6 DoFs (3 linear + 3 revolute), a prismatic/revolute joint 1 DoF, a fixed joint 0
        DoF, etc.

        Returns:
            np.array[float[n]]: generalized coordinates.
        """

    @overload
    def getBaseOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the base orientation (expressed as a quaternion [w,x,y,z]).

        Returns:
            np.array[float[4]]: base orientation (expressed as a quaternion [w,x,y,z])
        """

    @overload
    def getBaseOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the base orientation (expressed as a rotation matrix).

        Returns:
            np.array[float[3,3]]: rotation matrix
        """

    def getGeneralizedVelocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the generalized velocities of the system.

        The dimension of the returned vector is equal to the number of degrees of freedom that each joint provides.
        A floating joint provides 6 DoFs (3 linear + 3 revolute), a prismatic/revolute joint 1 DoF, a fixed joint 0
        DoF, etc.

        Returns:
            np.array[float[n]]: generalized velocities.
        """

    def getGeneralizedAcceleration(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the generalized accelerations of the system.

        Returns:
            np.array[float[n]]: generalized accelerations.
        """

    def updateKinematics(self) -> None:
        """
        Update the kinematics.

        It is unnecessary to call this function if you are simulating your system. `integrate1` calls this function.
        Call this function if you want to get kinematic properties but you don't want to integrate.
        """

    @overload
    def setGeneralizedCoordinate(self, coordinates: "std::initializer_list<double>") -> None:
        """
        Set the generalized coordinates.

        Args:
            coordinates (list[float]): generalized coordinates to set.
        """

    @overload
    def setGeneralizedCoordinate(self, coordinates: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Set the generalized coordinates.

        Args:
            coordinates (np.array[float[n]]): generalized coordinates to set.
        """

    @overload
    def setGeneralizedVelocity(self, velocities: "std::initializer_list<double>") -> None:
        """
        Set the generalized velocities.

        Args:
            velocities (list[float]): generalized velocities to set.
        """

    @overload
    def setGeneralizedVelocity(self, velocities: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Set the generalized velocities.

        Args:
            velocities (np.array[float[n]]): generalized velocities to set.
        """

    @overload
    def setGeneralizedForce(self, forces: "std::initializer_list<double>") -> None:
        """
        Set the generalized forces.

        These are the feedforward generalized forces. In the PD control mode, this differs from the actual
        generalizedForces. The dimension should be the same as the number of DoFs.

        Args:
            forces (list[float]): generalized forces to set.
        """

    @overload
    def setGeneralizedForce(self, forces: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Set the generalized forces.

        These are the feedforward generalized forces. In the PD control mode, this differs from the actual
        generalizedForces. The dimension should be the same as the number of DoFs.

        Args:
            forces (np.array[float[n]]): generalized forces to set.
        """

    def getState(self) -> tuple:
        """
        Get the joint states.

        Returns:
            np.array[float[n]]: generalized coordinates.
            np.array[float[n]]: generalized velocities.
        """

    def setState(self, coordinates: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], velocities: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Set the joint states.

        Args:
            coordinates (np.array[float[n]]): generalized coordinates to set.
            velocities (np.array[float[n]]): generalized velocities to set.
        """

    def getGeneralizedForce(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the generalized forces.

        Returns:
            np.array[float[n]]: generalized forces.
        """

    def getFeedForwardGeneralizedForce(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the feedforward generalized forces.

        Returns:
            np.array[float[n]]: feedforward generalized forces.
        """

    def getMassMatrix(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        r"""
        Get the mass (inertia) matrix which is present in the dynamic equation of motion:

        .. math:: H(q) \ddot{q} + N(q, \dot{q}) = \tau

        where :math:`H(q)` is the mass inertia matrix, :math:`N(q, \dot{q})` are the non-linear terms, and
        :math:`\tau` are the forces/torques.

        Returns:
            np.array[float[n,n]]: mass inertia matrix.
        """

    def getNonlinearities(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        r"""
        Get the non linearity terms that are present in the dynamic equation of motion:

        .. math:: H(q) \ddot{q} + N(q, \dot{q}) = \tau

        where :math:`N(q, \dot{q})` are the non-linear terms, :math:`H(q)` is the inertia matrix, and :math:`\tau` are
        the forces/torques.

        Returns:
            np.array[float[n]]: non-linearity forces.
        """

    def getInverseMassMatrix(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        r"""
        Get the inverse of the mass (inertia) matrix, where the inertia matrix appears in the dynamic equation of motion:

        .. math:: H(q) \ddot{q} + N(q, \dot{q}) = \tau

        where :math:`H(q)` is the mass inertia matrix, :math:`N(q, \dot{q})` are the non-linear terms, and
        :math:`\tau` are the forces/torques.

        Returns:
            np.array[float[n,n]]: inverse of the mass inertia matrix.
        """

    def getCompositeCOM(self) -> list:
        """
        Get the center of mass position of the composite body (i.e. the byproduct of the composite rigid body algorithm).

        Returns:
            list of np.array[float[3]]: center of mass position of the composite bodies.
        """

    def getCompositeInertia(self) -> list:
        """
        Get the composite moment of inertia of the composite body (i.e. the byproduct of the composite rigid body algorithm).

        Returns:
            list of np.array[float[3, 3]]: moment inertia of the composite bodies
        """

    def getCompositeMass(self) -> list[float]:
        """
        Get the composite mass of the composite body (i.e. the byproduct of the composite rigid body algorithm)

        Returns:
            list of float64: mass of the composite bodies
        """

    def getLinearMomentum(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the linear momentum of the whole system in Cartesian space.

        Returns:
            np.array[float[3]]: total linear momentum.
        """

    def getGeneralizedMomentum(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        r"""
        Get the generalized momentum which is simply the mass matrix multiplied by the generalized velocities:

        .. math:: H(q) \dot{q}

        where :math:`H(q)` is the mass/inertia matrix, and :math:`\dot{q}` are the generalized velocities.

        Returns:
            np.array[float[n]]: generalized momentum.
        """

    def get_kinetic_energy(self) -> float:
        """Return the total kinetic energy of the whole system."""

    def getPotentialEnergy(self, gravity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> float:
        """
        Get the system's potential energy due to gravity.

        Args:
            gravity (np.array[float[3]]): gravity vector.

        Returns:
            float: potential energy.
        """

    def getEnergy(self, gravity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> float:
        """
        Get the system's total energy.

        Args:
            gravity (np.array[float[3]]): gravity vector.

        Returns:
            float: total energy.
        """

    def printOutBodyNamesInOrder(self) -> None:
        """Print the moving bodies in the order. Fixed bodies are optimized out."""

    def printOutFrameNamesInOrder(self) -> None:
        """
        Print the frames (that are attached to every joint coordinate) in the order.
        """

    @overload
    def getPosition(self, body_idx: int, body_point: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the body's position with respect to the world frame.

        Args:
            body_idx (int): body/link index.
            body_point (np.array[float[3]]): position of point on the body expressed in the body frame.

        Returns:
            np.array[float[3]]: position of the body point in the world frame.
        """

    @overload
    def getPosition(self, frame_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the joint frame's position with respect to the world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: position of the joint frame expressed in the world frame.
        """

    def getFrameByName(self, name: str) -> CoordinateFrame:
        """
        Get the coordinate frame from its name.

        Args:
            name (str): name of the frame.

        Returns:
            CoordinateFrame: the coordinate frame.
        """

    def getFrameByIdx(self, idx: int) -> CoordinateFrame:
        """
        Get the coordinate frame from its index.

        Args:
            idx (int): index of the frame.

        Returns:
            CoordinateFrame: the coordinate frame.
        """

    def getSprings(self) -> list[SpringElement]:
        """
        Get the spring elements.

        Returns:
            springs: list of spring elements.
        """

    def getParentVector(self) -> list[int]:
        """
        Get the parent vector. parent[i] is the parent body id of body i.

        Returns:
            list[int]: parent body indices.
        """

    def getFrameIdxByName(self, name: str) -> int:
        """
        Get the coordinate frame index from its name.

        Args:
            name (str): name of the frame.

        Returns:
            int: the corresponding index.
        """

    def getFrames(self) -> list[CoordinateFrame]:
        """
        Get all the coordinate frames.

        Returns:
            list[CoordinateFrame]: the coordinate frames.
        """

    def getSensorSet(self, name: str) -> SensorSet:
        """
        Get the sensor set by name.

        Args:
            name (str): sensor set name.

        Returns:
            SensorSet: sensor set.
        """

    def getSensorSets(self) -> list[SensorSet]:
        """
        Get all sensor sets.

        Returns:
            list[SensorSet]: sensor sets.
        """

    @overload
    def getFramePosition(self, frame_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the frame position expressed in the Cartesian world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: the coordinate frame position in the world space.
        """

    @overload
    def getFramePosition(self, frame_name: str) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the frame position expressed in the Cartesian world frame.

        Args:
            frame_name (string): frame name.

        Returns:
            np.array[float[3]]: the coordinate frame position in the world space.
        """

    @overload
    def getFrameOrientation(self, frame_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the frame orientation as a rotation matrix expressed in the Cartesian world frame

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3,3]]: the coordinate frame orientation in the world space.
        """

    @overload
    def getFrameOrientation(self, frame_name: str) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the frame orientation as a rotation matrix expressed in the Cartesian world frame

        Args:
            frame_name (string): frame name.

        Returns:
            np.array[float[3,3]]: the coordinate frame orientation (quaternion) in the world space.
        """

    @overload
    def getFrameOrientation(self, frame_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the joint frame's orientation (expressed as a rotation matrix) with respect to the world frame.

        Args:
            frame_id (int): frame index.

        Returns:
            np.array[float[3]]: orientation (rotation matrix) of the joint frame expressed in the world frame.
        """

    @overload
    def getFrameOrientation(self, frame_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the joint frame's orientation (expressed as a quaternion [w,x,y,z]) with respect to the world frame.

        Args:
            frame_id (int): frame index.

        Returns:
            np.array[float[3]]: orientation (quaternion) of the joint frame expressed in the world frame.
        """

    def getFrameVelocity(self, frame_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the frame linear velocity expressed in the Cartesian world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: the coordinate frame linear velocity in the world space.
        """

    def getFrameAngularVelocity(self, frame_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the frame angular velocity expressed in the Cartesian world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: the coordinate frame angular velocity in the world space.
        """

    @overload
    def getFrameAcceleration(self, frame_name: str) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the frame linear acceleration expressed in the Cartesian world frame.

        Args:
            frame_name (string): frame name.

        Returns:
            np.array[float[3]]: the coordinate frame linear acceleration in the world space.
        """

    @overload
    def getFrameAcceleration(self, frame: CoordinateFrame) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the frame linear acceleration expressed in the Cartesian world frame.

        Args:
            frame (CoordinateFrame): coordinate frame.

        Returns:
            np.array[float[3]]: the coordinate frame linear acceleration in the world space.
        """

    @overload
    def getVelocity(self, frame_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the joint frame's linear velocity with respect to the world frame.

        Args:
            frame_id (int): frame id.

        Returns:
            np.array[float[3]]: velocity of the joint frame expressed in the world frame.
        """

    @overload
    def getVelocity(self, sparse_linear_jacobian: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Return the velocity of the point of the sparse linear jacobian.

        Args:
            sparse_linear_jacobian (np.array[float[3,n]]): sparse linear jacobian.

        Returns:
            np.array[float[3]]: velocity of the point expressed in the world frame.
        """

    @overload
    def getVelocity(self, body_id: int, body_pos: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Return the velocity of a point (expressed in the body frame) in the world frame.

        Args:
            body_id (int): body id.
            body_pos (np.array[float[3]]): position of a point on the body frame.

        Returns:
            np.array[float[3]]: velocity of the body expressed in the world frame.
        """

    def getAngularVelocity(self, body_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the angular velocity of the body with respect to the world frame.

        Args:
            body_id (int): body id.

        Returns:
            np.array[float[3]]: angular velocity of the body expressed in the world frame.
        """

    def getDenseFrameJacobian(self, frame_name: str) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        r"""
        Get the dense frame linear jacobian.

        .. math:: v_{lin} = J(q) \dot{q}

        Args:
            frame_name (string): frame name defined in the URDF. A frame is attached to every joint by default.

        Returns:
            np.array[float[3, n]]: dense linear jacobian of the frame.
        """

    def getDenseFrameRotationalJacobian(self, body_idx: str) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        r"""
        Get the dense rotational jacobian.

        .. math:: \omega = J(q) \dot{q}

        Args:
            body_idx (int): body index.

        Returns:
            np.array[float[3, n]]: dense rotational jacobian of the frame.
        """

    def getBodyIdx(self, name: str) -> int:
        """
        Return the body index associated with the given name.

        Args:
            name (str): body name.

        Returns:
            int: body index.
        """

    def getDOF(self) -> int:
        """
        Return the number of degrees of freedom.

        Returns:
            int: the number of degrees of freedom.
        """

    def getGeneralizedCoordinateDim(self) -> int:
        """
        Return the dimension/size of the generalized coordinates vector.

        Returns:
            int: the number of generalized coordinates.
        """

    def getBodyPosition(self, body_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Return the body position.

        Args:
            body_id (int): body id.
        Returns:
            np.array[float[3]]: body position.
        """

    def getBodyOrientation(self, body_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Return the body orientation.

        Args:
            body_id (int): body id.
        Returns:
            np.array[float[3,3]]: body orientation.
        """

    @property
    def getJointPos_P(self) -> list: ...

    @getJointPos_P.setter
    def getJointPos_P(self, arg: list, /) -> None: ...

    def setJointPos_P(self, positions: list) -> None:
        """
        Set the joint Cartesian positions relative to their parent frame.

        Args:
            positions (list[np.array[float[3]]]): joint cartesian positions (relative to their parent frame).
        """

    @overload
    def getMass(self) -> list[float]:
        """
        Return the body/link masses.

        Returns:
            list[double]: body masses.
        """

    @overload
    def getMass(self, localIdx: int) -> float:
        """
        Get the mass of the link.

        Args:
            link_idx (int): link index.

        Returns:
            float: mass value.
        """

    @overload
    def setMass(self, masses: Sequence[float]) -> None:
        """
        Set the body/link masses.

        Args:
            masses (list[double]): body masses.
        """

    @overload
    def setMass(self, link_idx: int, value: float) -> None:
        """
        Set the mass of the link.

        Args:
            link_idx (int): link index.
            value (float): mass value.
        """

    @property
    def masses(self) -> list[float]: ...

    @masses.setter
    def masses(self, arg: Sequence[float], /) -> None: ...

    def getInertia(self) -> list:
        """
        Return the inertias (one for each body).

        Returns:
            list[np.array[float[3,3]]]: inertias.
        """

    def setInertia(self, inertias: list) -> None:
        """
        Set the inertias (one for each body).

        Args:
            inertias (list[np.array[float[3,3]]]): body inertias.
        """

    @property
    def inertias(self) -> list: ...

    @inertias.setter
    def inertias(self, arg: list, /) -> None: ...

    def getBodyCOM_B(self) -> list:
        """
        Return the center of mass of each link (expressed in their body frame).

        Returns:
            list[np.array[float[3]]]: center of mass of each link (expressed in the body frame).
        """

    def setLinkCOM(self, coms: list) -> None:
        """
        Set the center of mass of each link (expressed in their body frame).

        Args:
            coms (list[np.array[float[3]]]): center of mass of each link (expressed in the body frame).
        """

    @property
    def linkComs(self) -> list: ...

    @linkComs.setter
    def linkComs(self, arg: list, /) -> None: ...

    def updateMassInfo(self) -> None:
        """
        Update the mass information. This function must be called after we change the dynamic parameters.
        """

    def getTotalMass(self) -> float:
        """
        Get the total mass of the system.

        Returns:
            float: total mass value.
        """

    @overload
    def setExternalForce(self, local_idx: int, pos: Annotated[NDArray[numpy.float64], dict(order='C')], force: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the external force on the body.

        Args:
            local_idx (int): local/link index.
            force (np.array[float[3]]): force vector.
        """

    @overload
    def setExternalForce(self, local_idx: int, force_frame: ArticulatedSystem.Frame, force: Annotated[NDArray[numpy.float64], dict(order='C')], pos_frame: ArticulatedSystem.Frame, position: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the external force on the specified point on the body.

        Args:
            local_idx (int): local/link index.
            force_frame (Frame): frame in which the force vector is expressed in, select between {WORLD_FRAME,
                BODY_FRAME, PARENT_FRAME}.
            force (np.array[float[3]]): force vector.
            pos_frame (Frame): frame in which the position vector is expressed in, select between {WORLD_FRAME,
                BODY_FRAME, PARENT_FRAME}.
            position (np.array[float[3]]): position vector.
        """

    def setExternalTorque(self, local_idx: int, torque: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the external force on the body.

        Args:
            local_idx (int): local/link index.
            force (np.array[float[3]]): force vector.
        """

    def setControlMode(self, mode: ControlMode) -> None:
        """
        Set the control mode.

        Args:
            mode (ControlMode.Type): control mode type, select between {FORCE_AND_TORQUE, PD_PLUS_FEEDFORWARD_TORQUE,
                VELOCITY_PLUS_FEEDFORWARD_TORQUE}
        """

    def getControlMode(self) -> ControlMode:
        """
        Get the control mode.

        Returns:
            ControlMode.Type: control mode type which is one of {FORCE_AND_TORQUE, PD_PLUS_FEEDFORWARD_TORQUE,
                VELOCITY_PLUS_FEEDFORWARD_TORQUE}
        """

    def setPdTarget(self, pos_targets: Annotated[NDArray[numpy.float64], dict(order='C')], vel_targets: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the PD targets. It is effective only in the control mode 'PD_PLUS_FEEDFORWARD_TORQUE'. Set any arbitrary
                number for unactuated degrees of freedom.

        	    Args:
        	        pos_targets (np.array[float[n]]): position targets.
        	        vel_targets (np.array[float[n]]): velocity targets.
        """

    def setPdGains(self, p_gains: Annotated[NDArray[numpy.float64], dict(order='C')], d_gains: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the PD gains. It is effective only in the control mode 'PD_PLUS_FEEDFORWARD_TORQUE'. Set any arbitrary
                number for unactuated degrees of freedom.

        	    Args:
        	        p_gains (np.array[float[n]]): P gains.
        	        d_gains (np.array[float[n]]): D gains.
        """

    def setJointDamping(self, dampings: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Set the joint dampings (passive elements at the joints).

        Args:
            dampings (np.array[float[n]]): joint damping coefficients.
        """

    def computeSparseInverse(self, mass_matrix: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        This computes the inverse mass matrix given the mass matrix. The return type is dense. It exploits the
        sparsity of the mass matrix to efficiently perform the computation.

        Args:
            mass_matrix (np.array[float[n,n]]): mass matrix.

        Returns:
            np.array(float[n,n]): dense inverse matrix.
        """

    def massMatrixVecMul(self, vector: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def ignoreCollisionBetween(self, body_idx1: int, body_idx2: int) -> None:
        """
        Ignore collision between the 2 specified bodies.

        Args:
            body_idx1 (int): first body index.
            body_idx2 (int): second body index.
        """

    def getOptions(self) -> ArticulatedSystemOption:
        """
        Return the options associated with the articulated system.

        Returns:
            ArticulatedSystemOption: options for the articulated system.
        """

    def getBodyNames(self) -> list[str]:
        """
        Return the body names.

        Returns:
            list[str]: list of body names.
        """

    def getVisOb(self) -> list[VisObject]:
        """
        Get the visual objects.

        Returns:
            list[VisObject]: list of visual objects.
        """

    def getVisColOb(self) -> list[VisObject]:
        """
        Get the visual collision objects.

        Returns:
            list[VisObject]: list of visual collision objects.
        """

    def getVisualObjectPosition(self, arg: int, /) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the visual object position.

        Args:
            body_idx (int): body index.

        Returns:
            np.array[float[3]]: visual object position.
        """

    def getVisualObjectOrientation(self, arg: int, /) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the visual object orientation in quaternion.

        Args:
            body_idx (int): body index.

        Returns:
            np.array[float[4]]: visual object orientation (expressed as a quaternion [w,x,y,z]).
        """

    def getVisColObPosition(self, arg: int, /) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the visual collision object position.

        Args:
            body_idx (int): body index.

        Returns:
            np.array[float[3]]: visual object position.
        """

    def getVisColObOrientation(self, arg: int, /) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the visual collision object orientation in a quaternion.

        Args:
            body_idx (int): body index.

        Returns:
            np.array[float[4]]: visual object orientation (expressed as a quaternion [w,x,y,z]).
        """

    def getResourceDir(self) -> str:
        """
        Get the robot resource directory.

        Returns:
            str: robot resource directory.
        """

    def getRobotDescriptionfFileName(self) -> str:
        """
        Get the robot description filename (e.g. path to the URDF).

        Returns:
            str: robot description filename.
        """

    def getRobotDescriptionfTopDirName(self) -> str:
        """
        Get the robot description top directory name.

        Returns:
            str: robot description top directory name.
        """

    def setBasePos(self, position: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the base position.

        Args:
            position (np.array[float[3]]): new base position.
        """

    def setBaseOrientation(self, orientation: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the base orientation.

        Args:
            orientation (np.array[float[3]], np.array[float[4]], np.array[float[3,3]]): new base orientation.
        """

    def setActuationLimits(self, upper: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], lower: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Set the upper and lower limits in actuation forces.

        Args:
            upper (np.array[float[n]]): upper limits.
            lower (np.array[float[n]]): lower limits.
        """

    def setComputeInverseDynamics(self, flag: bool) -> None:
        """
        Enable or disable inverse dynamics computation.

        Args:
            flag (bool): true to enable inverse dynamics.
        """

    def getComputeInverseDynamics(self) -> bool:
        """
        Check if inverse dynamics computation is enabled.

        Returns:
            bool: true if inverse dynamics is enabled.
        """

    def getForceAtJointInWorldFrame(self, joint_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the force at the specified joint in the world frame.

        Args:
            joint_id (int): joint id.

        Returns:
            np.array[float[3]]: joint force in world frame.
        """

    def getTorqueAtJointInWorldFrame(self, joint_id: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the torque at the specified joint in the world frame.

        Args:
            joint_id (int): joint id.

        Returns:
            np.array[float[3]]: joint torque in world frame.
        """

    def getAllowedNumberOfInternalContactsBetweenTwoBodies(self) -> int:
        """
        Get the allowed number of internal contacts between two bodies.

        Returns:
            int: maximum number of internal contacts.
        """

    def setAllowedNumberOfInternalContactsBetweenTwoBodies(self, count: int) -> None:
        """
        Set the allowed number of internal contacts between two bodies.

        Args:
            count (int): maximum number of internal contacts.
        """

    def isSecondOrderOrHigher(self, scheme: ArticulatedSystem.IntegrationScheme) -> bool: ...

    def getBasePosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def setBasePos_e(self, position: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def setBaseOrientation_e(self, orientation: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def setBaseVelocity(self, velocity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def setBaseAngularVelocity(self, velocity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def getBodyPose(self, body_idx: int) -> tuple: ...

    def getBodyCOM_W(self) -> list: ...

    def getCOM(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getKineticEnergy(self) -> float: ...

    def getAngularMomentum(self, reference_point: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getGeneralizedVelocityDim(self) -> int: ...

    def getGeneralizedVelocityIndex(self, name: str) -> int: ...

    def getActuationUpperLimits(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getActuationLowerLimits(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getJointLimits(self) -> list: ...

    def setJointLimits(self, joint_limits: list) -> None: ...

    def getJointVelocityLimits(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def setJointVelocityLimits(self, velocity_limits: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def getRotorInertia(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def setRotorInertia(self, rotor_inertia: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def getNumberOfJoints(self) -> int: ...

    def getJointType(self, joint_index: int) -> JointType: ...

    def getJointAxis(self, joint_index: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getJointAxis_P(self) -> list: ...

    def getMappingFromBodyIndexToGeneralizedCoordinateIndex(self) -> list[int]: ...

    def getMappingFromBodyIndexToGeneralizedVelocityIndex(self) -> list[int]: ...

    def getJoint(self, name: str) -> JointRef: ...

    def getLink(self, name: str) -> LinkRef: ...

    def getJointLimitViolations(self, contact_problems: Sequence[contact.Single3DContactProblem]) -> list[contact.Single3DContactProblem]: ...

    def getFrameByLinkName(self, name: str) -> CoordinateFrame: ...

    def getFrameIdxByLinkName(self, name: str) -> int: ...

    def getPositionInBodyCoordinate(self, body_idx: int, pos_w: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getPositionInFrame(self, frame_id: int, local_pos: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @overload
    def getSparseJacobian(self, body_idx: int, point_w: Annotated[NDArray[numpy.float64], dict(order='C')]) -> SparseJacobian: ...

    @overload
    def getSparseJacobian(self, body_idx: int, frame: ArticulatedSystem.Frame, point: Annotated[NDArray[numpy.float64], dict(order='C')]) -> SparseJacobian: ...

    def getSparseRotationalJacobian(self, body_idx: int) -> SparseJacobian: ...

    def getTimeDerivativeOfSparseJacobian(self, body_idx: int, frame: ArticulatedSystem.Frame, point: Annotated[NDArray[numpy.float64], dict(order='C')]) -> SparseJacobian: ...

    def getTimeDerivativeOfSparseRotationalJacobian(self, body_idx: int) -> SparseJacobian: ...

    @staticmethod
    def convertSparseJacobianToDense(sparse_jacobian: SparseJacobian, cols: int = 0) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    def getDenseJacobian(self, body_idx: int, point_w: Annotated[NDArray[numpy.float64], dict(order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    def getDenseRotationalJacobian(self, body_idx: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    def getUdot(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getMinvJT(self) -> list: ...

    def getj_MinvJT_T1D(self) -> list: ...


    def getVisObPose(self, idx: int) -> tuple: ...

    def getVisColObPose(self, idx: int) -> tuple: ...

    def exportRobotDescriptionToURDF(self, file_path: str) -> None: ...

    def getRobotDescriptionFullPath(self) -> str: ...

    def getRobotDescription(self) -> str: ...

    def setIntegrationScheme(self, scheme: ArticulatedSystem.IntegrationScheme) -> None: ...

    def setExternalTorqueInBodyFrame(self, body_idx: int, torque: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def addSpring(self, spring: SpringElement) -> None: ...

    def addConstraints(self, pin_def: Sequence[constraints.PinConstraintDefinition], pin_constraint_nominal_config: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def initializeConstraints(self) -> None: ...

    def getIgnoredCollisionPairs(self) -> list: ...

    def updateSelfCollisionCache(self, world: World) -> None: ...

    def printOutMovableJointNamesInOrder(self) -> None: ...

    def getMovableJointNames(self) -> list[str]: ...

    def getCollisionBodies(self) -> list[CollisionDefinition]: ...

    def getCollisionBody(self, name: str) -> CollisionDefinition: ...

    def setCollisionBodyShapeParameters(self, id_: int, parameters: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def setCollisionBodyPositionOffset(self, id_: int, position: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def setCollisionBodyOrientationOffset(self, id_: int, orientation: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def setCollisionObjectShapeParameters(self, id_: int, parameters: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the collision object shape parameters.

        Args:
            id_ (int): collision object id.
            parameters (list[float]): parameters.
        """

    def setCollisionObjectPositionOffset(self, id_: int, position: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the collision object position offset.

        Args:
            id_ (int): collision object id.
            position (np.array[float[3]]): position offset.
        """

    def setCollisionObjectOrientationOffset(self, id_: int, orientation: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the collision object orientation offset.

        Args:
            id_ (int): collision object id.
            orientation (np.array[float[3]], np.array[float[4]], np.array[float[3,3]]): orientation offset.
        """

    @overload
    def setBodySegmentationLabel(self, bodyIdx: int, label: int) -> None: ...

    @overload
    def setBodySegmentationLabel(self, body: str, label: int) -> None: ...

    def getBodySegmentationLabel(self, bodyIdx: int) -> int: ...

    def clearBodySegmentationLabels(self) -> None: ...

    class IKOptions:
        def __init__(self) -> None: ...

        @property
        def maxIterations(self) -> int: ...

        @maxIterations.setter
        def maxIterations(self, arg: int, /) -> None: ...

        @property
        def positionTolerance(self) -> float: ...

        @positionTolerance.setter
        def positionTolerance(self, arg: float, /) -> None: ...

        @property
        def orientationTolerance(self) -> float: ...

        @orientationTolerance.setter
        def orientationTolerance(self, arg: float, /) -> None: ...

        @property
        def orientationWeight(self) -> float: ...

        @orientationWeight.setter
        def orientationWeight(self, arg: float, /) -> None: ...

        @property
        def damping(self) -> float: ...

        @damping.setter
        def damping(self, arg: float, /) -> None: ...

        @property
        def stepSize(self) -> float: ...

        @stepSize.setter
        def stepSize(self, arg: float, /) -> None: ...

        @property
        def maxStep(self) -> float: ...

        @maxStep.setter
        def maxStep(self, arg: float, /) -> None: ...

        @property
        def svdTolerance(self) -> float: ...

        @svdTolerance.setter
        def svdTolerance(self, arg: float, /) -> None: ...

        @property
        def finiteDifferenceStep(self) -> float: ...

        @finiteDifferenceStep.setter
        def finiteDifferenceStep(self, arg: float, /) -> None: ...

        @property
        def useOrientation(self) -> bool: ...

        @useOrientation.setter
        def useOrientation(self, arg: bool, /) -> None: ...

        @property
        def applySolution(self) -> bool: ...

        @applySolution.setter
        def applySolution(self, arg: bool, /) -> None: ...

        @property
        def applyBestOnFailure(self) -> bool: ...

        @applyBestOnFailure.setter
        def applyBestOnFailure(self, arg: bool, /) -> None: ...

        @property
        def enforceJointLimits(self) -> bool: ...

        @enforceJointLimits.setter
        def enforceJointLimits(self, arg: bool, /) -> None: ...

        @property
        def inverseMethod(self) -> str: ...

        @inverseMethod.setter
        def inverseMethod(self, arg: str, /) -> None: ...

        @property
        def transposeGain(self) -> float: ...

        @transposeGain.setter
        def transposeGain(self, arg: float, /) -> None: ...

        @property
        def seed(self) -> list[float]: ...

        @seed.setter
        def seed(self, arg: Sequence[float], /) -> None: ...

    class IKResult:
        @property
        def success(self) -> bool: ...

        @property
        def iterations(self) -> int: ...

        @property
        def bestIteration(self) -> int: ...

        @property
        def residual(self) -> float: ...

        @property
        def positionResidual(self) -> float: ...

        @property
        def orientationResidual(self) -> float: ...

        @property
        def bestResidual(self) -> float: ...

        @property
        def bestPositionResidual(self) -> float: ...

        @property
        def bestOrientationResidual(self) -> float: ...

        @property
        def solution(self) -> list[float]: ...

        @property
        def bestSolution(self) -> list[float]: ...

        @property
        def bestFramePosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @property
        def bestFrameOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @property
        def message(self) -> str: ...

    @overload
    def solveIK(self, frame: str, targetPosition: Annotated[NDArray[numpy.float64], dict(order='C')], options: ArticulatedSystem.IKOptions = ...) -> ArticulatedSystem.IKResult: ...

    @overload
    def solveIK(self, frame: int, targetPosition: Annotated[NDArray[numpy.float64], dict(order='C')], options: ArticulatedSystem.IKOptions = ...) -> ArticulatedSystem.IKResult: ...

    @overload
    def solveIK(self, frame: str, targetPosition: Annotated[NDArray[numpy.float64], dict(order='C')], targetOrientation: Annotated[NDArray[numpy.float64], dict(order='C')], options: ArticulatedSystem.IKOptions = ...) -> ArticulatedSystem.IKResult: ...

    @overload
    def solveIK(self, frame: int, targetPosition: Annotated[NDArray[numpy.float64], dict(order='C')], targetOrientation: Annotated[NDArray[numpy.float64], dict(order='C')], options: ArticulatedSystem.IKOptions = ...) -> ArticulatedSystem.IKResult: ...

    def setActuators(self, actuators: Sequence[ActuatorDefinition]) -> None: ...

    def getNumberOfActuators(self) -> int: ...

    def getActuatorNames(self) -> list[str]: ...

    def getActuators(self) -> list[ActuatorDefinition]: ...

    def getActuatorIndex(self, name: str) -> int: ...

    def setActuatorTorques(self, torques: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None: ...

    @overload
    def setActuatorTorque(self, name: str, torque: float) -> None: ...

    @overload
    def setActuatorTorque(self, index: int, torque: float) -> None: ...

    def getActuatorTorques(self) -> list[float]: ...

    def getActuatorTorqueBounds(self) -> tuple: ...

    @overload
    def setBusVoltage(self, voltage: float) -> None: ...

    @overload
    def setBusVoltage(self, actuatorName: str, voltage: float) -> None: ...

    def getActuatorStates(self) -> list[ActuatorState]: ...

    def setMotorOperatingRegionEnforced(self, enforce: bool) -> None: ...

    def isMotorOperatingRegionEnforced(self) -> bool: ...

    def getTracks(self) -> list[Track]: ...

    def getTrackIndex(self, name: str) -> int: ...

    def getTrackSpeed(self, index: int) -> float: ...

class IKOptions:
    def __init__(self) -> None: ...

    @property
    def maxIterations(self) -> int: ...

    @maxIterations.setter
    def maxIterations(self, arg: int, /) -> None: ...

    @property
    def positionTolerance(self) -> float: ...

    @positionTolerance.setter
    def positionTolerance(self, arg: float, /) -> None: ...

    @property
    def orientationTolerance(self) -> float: ...

    @orientationTolerance.setter
    def orientationTolerance(self, arg: float, /) -> None: ...

    @property
    def orientationWeight(self) -> float: ...

    @orientationWeight.setter
    def orientationWeight(self, arg: float, /) -> None: ...

    @property
    def damping(self) -> float: ...

    @damping.setter
    def damping(self, arg: float, /) -> None: ...

    @property
    def stepSize(self) -> float: ...

    @stepSize.setter
    def stepSize(self, arg: float, /) -> None: ...

    @property
    def maxStep(self) -> float: ...

    @maxStep.setter
    def maxStep(self, arg: float, /) -> None: ...

    @property
    def svdTolerance(self) -> float: ...

    @svdTolerance.setter
    def svdTolerance(self, arg: float, /) -> None: ...

    @property
    def finiteDifferenceStep(self) -> float: ...

    @finiteDifferenceStep.setter
    def finiteDifferenceStep(self, arg: float, /) -> None: ...

    @property
    def useOrientation(self) -> bool: ...

    @useOrientation.setter
    def useOrientation(self, arg: bool, /) -> None: ...

    @property
    def applySolution(self) -> bool: ...

    @applySolution.setter
    def applySolution(self, arg: bool, /) -> None: ...

    @property
    def applyBestOnFailure(self) -> bool: ...

    @applyBestOnFailure.setter
    def applyBestOnFailure(self, arg: bool, /) -> None: ...

    @property
    def enforceJointLimits(self) -> bool: ...

    @enforceJointLimits.setter
    def enforceJointLimits(self, arg: bool, /) -> None: ...

    @property
    def inverseMethod(self) -> str: ...

    @inverseMethod.setter
    def inverseMethod(self, arg: str, /) -> None: ...

    @property
    def transposeGain(self) -> float: ...

    @transposeGain.setter
    def transposeGain(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> list[float]: ...

    @seed.setter
    def seed(self, arg: Sequence[float], /) -> None: ...

class IKResult:
    @property
    def success(self) -> bool: ...

    @property
    def iterations(self) -> int: ...

    @property
    def bestIteration(self) -> int: ...

    @property
    def residual(self) -> float: ...

    @property
    def positionResidual(self) -> float: ...

    @property
    def orientationResidual(self) -> float: ...

    @property
    def bestResidual(self) -> float: ...

    @property
    def bestPositionResidual(self) -> float: ...

    @property
    def bestOrientationResidual(self) -> float: ...

    @property
    def solution(self) -> list[float]: ...

    @property
    def bestSolution(self) -> list[float]: ...

    @property
    def bestFramePosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @property
    def bestFrameOrientation(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @property
    def message(self) -> str: ...

class DcMotorParameters:
    def __init__(self) -> None: ...

    @property
    def resistance(self) -> float: ...

    @resistance.setter
    def resistance(self, arg: float, /) -> None: ...

    @property
    def torqueConstant(self) -> float: ...

    @torqueConstant.setter
    def torqueConstant(self, arg: float, /) -> None: ...

    @property
    def velocityConstant(self) -> float: ...

    @velocityConstant.setter
    def velocityConstant(self, arg: float, /) -> None: ...

    @property
    def busVoltage(self) -> float: ...

    @busVoltage.setter
    def busVoltage(self, arg: float, /) -> None: ...

    @property
    def peakTorque(self) -> float: ...

    @peakTorque.setter
    def peakTorque(self, arg: float, /) -> None: ...

    def isValid(self) -> bool: ...

    def torquePerVolt(self) -> float: ...

    def backEmfDamping(self) -> float: ...

    def stallTorque(self) -> float: ...

    def noLoadSpeed(self) -> float: ...

class MotorSaturation(enum.Enum):
    NONE = 0

    PEAK_TORQUE = 1

    VOLTAGE = 2

class MotorTorqueBounds:
    @property
    def lower(self) -> float: ...

    @property
    def upper(self) -> float: ...

    @property
    def lowerSaturation(self) -> MotorSaturation: ...

    @property
    def upperSaturation(self) -> MotorSaturation: ...

class ActuatorCoupling:
    def __init__(self) -> None: ...

    @property
    def joint(self) -> str: ...

    @joint.setter
    def joint(self, arg: str, /) -> None: ...

    @property
    def ratio(self) -> float: ...

    @ratio.setter
    def ratio(self, arg: float, /) -> None: ...

class ActuatorDefinition:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def model(self) -> str: ...

    @model.setter
    def model(self, arg: str, /) -> None: ...

    @property
    def joint(self) -> str: ...

    @joint.setter
    def joint(self, arg: str, /) -> None: ...

    @property
    def gearRatio(self) -> float: ...

    @gearRatio.setter
    def gearRatio(self, arg: float, /) -> None: ...

    @property
    def couplings(self) -> list[ActuatorCoupling]: ...

    @couplings.setter
    def couplings(self, arg: Sequence[ActuatorCoupling], /) -> None: ...

    @property
    def motor(self) -> DcMotorParameters: ...

    @motor.setter
    def motor(self, arg: DcMotorParameters, /) -> None: ...

    @property
    def outputDamping(self) -> float: ...

    @outputDamping.setter
    def outputDamping(self, arg: float, /) -> None: ...

    @property
    def outputFriction(self) -> float: ...

    @outputFriction.setter
    def outputFriction(self, arg: float, /) -> None: ...

    def isValid(self) -> bool: ...

class ActuatorState:
    @property
    def motorSpeed(self) -> float: ...

    @property
    def averageMotorSpeed(self) -> float: ...

    @property
    def commandedMotorTorque(self) -> float: ...

    @property
    def motorTorque(self) -> float: ...

    @property
    def jointLossTorque(self) -> float: ...

    @property
    def saturation(self) -> MotorSaturation: ...

def motorTorqueBounds(motor: DcMotorParameters, speed: float) -> MotorTorqueBounds: ...

def clipMotorTorque(motor: DcMotorParameters, speed: float, torque: float) -> float: ...

def motorOperatingRegionViolation(motor: DcMotorParameters, speed: float, torque: float) -> float: ...

class TrackDrive:
    @property
    def joint(self) -> str: ...

    @property
    def gain(self) -> float: ...

    @property
    def gvIndex(self) -> int: ...

    @property
    def bodyIdx(self) -> int: ...

class Track:
    @property
    def name(self) -> str: ...

    @property
    def link(self) -> str: ...

    @property
    def bodyIdx(self) -> int: ...

    @property
    def axis(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @property
    def width(self) -> float: ...

    @property
    def thickness(self) -> float: ...

    @property
    def drives(self) -> list[TrackDrive]: ...

    @property
    def collisionBodies(self) -> list[int]: ...

class TerrainProperties:
    """Raisim terrain properties"""

    @overload
    def __init__(self) -> None:
        """Initialize the terrain properties"""

    @overload
    def __init__(self, frequency: float = 0.1, x_size: float = 10.0, y_size: float = 10.0, z_scale: float = 2.0, x_samples: int = 100, y_samples: int = 100, fractal_octaves: int = 5, fractal_lacunarity: float = 2.0, fractal_gain: float = 0.5, step_size: float = 0, height_offset: float = 0, seed: int = 1) -> None:
        """
        Initialize the terrain properties.

        Args:
            frequency (float): frequency.
            x_size (float): the size in the x direction.
            y_size (float): the size in the y direction.
            z_scale (float): z scale.
            x_samples (int): the number of samples in x.
            y_samples (int): the number of samples in y.
            fractal_octaves (int): the number of fractal octaves.
            fractal_lacunarity (float): the lacunarity of fractals.
            fractal_gain (float): fractal gain.
            step_size (float): the step size.
            height_offset (float): the height offset.
            seed (int): the random seed.
        """

    @property
    def frequency(self) -> float: ...

    @frequency.setter
    def frequency(self, arg: float, /) -> None: ...

    @property
    def xSize(self) -> float: ...

    @xSize.setter
    def xSize(self, arg: float, /) -> None: ...

    @property
    def ySize(self) -> float: ...

    @ySize.setter
    def ySize(self, arg: float, /) -> None: ...

    @property
    def zScale(self) -> float: ...

    @zScale.setter
    def zScale(self, arg: float, /) -> None: ...

    @property
    def xSamples(self) -> int: ...

    @xSamples.setter
    def xSamples(self, arg: int, /) -> None: ...

    @property
    def ySamples(self) -> int: ...

    @ySamples.setter
    def ySamples(self, arg: int, /) -> None: ...

    @property
    def fractalOctaves(self) -> int: ...

    @fractalOctaves.setter
    def fractalOctaves(self, arg: int, /) -> None: ...

    @property
    def fractalLacunarity(self) -> float: ...

    @fractalLacunarity.setter
    def fractalLacunarity(self, arg: float, /) -> None: ...

    @property
    def fractalGain(self) -> float: ...

    @fractalGain.setter
    def fractalGain(self, arg: float, /) -> None: ...

    @property
    def stepSize(self) -> float: ...

    @stepSize.setter
    def stepSize(self, arg: float, /) -> None: ...

    @property
    def heightOffset(self) -> float: ...

    @heightOffset.setter
    def heightOffset(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

class Ground(SingleBodyObject):
    """Raisim Ground."""

    def __init__(self, height: float) -> None:
        """
        Initialize a ground instance.

        Args:
            height (float): height of the ground.
        """

    def getHeight(self) -> float:
        """
        Get the ground's height.

        Returns:
            float: height of the ground.
        """

class HeightMap(SingleBodyObject):
    """Raisim HeightMap."""

    @overload
    def __init__(self, x_center: float, y_center: float, filename: str) -> None:
        """
        Initialize a HeightMap instance.

        Args:
            x_center (float): the x center of the heightmap in the world.
            y_center (float): the y center of the heightmap in the world.
            filename (str): raisim heightmap filename.
        """

    @overload
    def __init__(self, x_center: float, y_center: float, filename: str, x_size: float, y_size: float, height_scale: float, height_offset: float) -> None:
        """
        Initialize a HeightMap instance.

        Args:
            x_center (float): the x center of the heightmap in the world.
            y_center (float): the y center of the heightmap in the world.
            filename (str): filename to the PNG.
            x_size (float): the size in the x direction.
            y_size (float): the size in the y direction.
            height_scale (float): the height scale.
            height_offset (float): the height offset.
        """

    @overload
    def __init__(self, x_samples: int, y_samples: int, x_scale: float, y_scale: float, x_center: float, y_center: float, heights: Sequence[float]) -> None:
        """
        Initialize a HeightMap instance.

        Args:
            x_samples (int): the number of samples in x.
            y_samples (int): the number of samples in y.
            x_scale (float): the scale in the x direction.
            y_scale (float): the scale in the y direction.
            x_center (float): the x center of the heightmap in the world.
            y_center (float): the y center of the heightmap in the world.
            heights (list[float]): list of desired heights.
        """

    @overload
    def __init__(self, x_center: float, y_center: float, terrain_properties: TerrainProperties) -> None:
        """
        Initialize a HeightMap instance.

        Args:
            x_center (float): the x center of the heightmap in the world.
            y_center (float): the y center of the heightmap in the world.
            terrain_properties (TerrainProperties): the terrain properties.
        """

    @overload
    def update(self, x_center: float, y_center: float, terrain_properties: TerrainProperties) -> None: ...

    @overload
    def update(self, x_center: float, y_center: float, x_size: float, y_size: float, height: Sequence[float]) -> None: ...

    def init(self, x_samples: int, y_samples: int, x_scale: float, y_scale: float, x_center: float, y_center: float) -> None:
        """
        Initialize the heightmap.

        	    Args:
        	        x_samples (int): the number of samples in x.
        	        y_samples (int): the number of samples in y.
                    x_scale (float): the scale in the x direction.
                    y_scale (float): the scale in the y direction.
                    x_center (float): the x center of the heightmap in the world.
                    y_center (float): the y center of the heightmap in the world.
        """

    def getHeightMap(self) -> list[float]: ...

    def getHeight(self, x: float, y: float) -> float:
        """
        Get the height at the given location.

        	    Args:
                    x (float): x position.
                    y (float): y position.
        """

    def getNormal(self, x: float, y: float) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the normal at the given location.

        	    Args:
                    x (float): x position.
                    y (float): y position.
        """

    def getXSamples(self) -> int:
        """
        Get the x samples.

        	    Returns:
                    int: the number of x samples.
        """

    def getYSamples(self) -> int:
        """
        Get the y samples.

        	    Returns:
                    int: the number of y samples.
        """

    def getXSize(self) -> float:
        """
        Get the x size.

        	    Returns:
                    float: the size in x.
        """

    def getYSize(self) -> float:
        """
        Get the y size.

        	    Returns:
                    float: the size in y.
        """

    def getCenterX(self) -> float:
        """
        Get the x coordinate of the grid center in the height map frame. It is the world x of
        	    the center only while the height map has the identity pose.

        	    Returns:
                    float: the x center in the height map frame.
        """

    def getCenterY(self) -> float:
        """
        Get the y coordinate of the grid center in the height map frame.

        	    Returns:
                    float: the y center in the height map frame.
        """

    def getHeightVector(self) -> list[float]:
        """
        Get the height vector.

        	    Returns:
                    list[float]: the height vector.
        """

    def getContactHeightVector(self) -> list[float]: ...

    def setUpdatedFalse(self) -> None: ...

    def isUpdated(self) -> bool: ...

    def setColor(self, color_level: Sequence[ColorRGB]) -> None: ...

    def getColorMap(self) -> list[ColorRGB]: ...

    def getColor(self, x: float, y: float) -> ColorRGB: ...

class DeformableObject(Object):
    class SolverMode(enum.Enum):
        XPBD = 0

        PBD = 1

    class Material:
        def __init__(self) -> None: ...

        @property
        def totalMass(self) -> float: ...

        @totalMass.setter
        def totalMass(self, arg: float, /) -> None: ...

        @property
        def distanceCompliance(self) -> float: ...

        @distanceCompliance.setter
        def distanceCompliance(self, arg: float, /) -> None: ...

        @property
        def distanceStiffness(self) -> float: ...

        @distanceStiffness.setter
        def distanceStiffness(self, arg: float, /) -> None: ...

        @property
        def bendCompliance(self) -> float: ...

        @bendCompliance.setter
        def bendCompliance(self, arg: float, /) -> None: ...

        @property
        def bendStiffness(self) -> float: ...

        @bendStiffness.setter
        def bendStiffness(self, arg: float, /) -> None: ...

        @property
        def youngsModulus(self) -> float: ...

        @youngsModulus.setter
        def youngsModulus(self, arg: float, /) -> None: ...

        @property
        def poissonRatio(self) -> float: ...

        @poissonRatio.setter
        def poissonRatio(self, arg: float, /) -> None: ...

        @property
        def thickness(self) -> float: ...

        @thickness.setter
        def thickness(self, arg: float, /) -> None: ...

        @property
        def crossSectionArea(self) -> float: ...

        @crossSectionArea.setter
        def crossSectionArea(self, arg: float, /) -> None: ...

        @property
        def damping(self) -> float: ...

        @damping.setter
        def damping(self, arg: float, /) -> None: ...

        @property
        def airDrag(self) -> float: ...

        @airDrag.setter
        def airDrag(self, arg: float, /) -> None: ...

        @property
        def collisionRadius(self) -> float: ...

        @collisionRadius.setter
        def collisionRadius(self, arg: float, /) -> None: ...

        @property
        def iterations(self) -> int: ...

        @iterations.setter
        def iterations(self, arg: int, /) -> None: ...

        @property
        def substeps(self) -> int: ...

        @substeps.setter
        def substeps(self, arg: int, /) -> None: ...

        @property
        def solverMode(self) -> DeformableObject.SolverMode: ...

        @solverMode.setter
        def solverMode(self, arg: DeformableObject.SolverMode, /) -> None: ...

        @property
        def selfCollision(self) -> bool: ...

        @selfCollision.setter
        def selfCollision(self, arg: bool, /) -> None: ...

    class InternalStrutOptions:
        def __init__(self) -> None: ...

        class Mode(enum.Enum):
            PairsWithinRadius = 1

        @property
        def mode(self) -> DeformableObject.InternalStrutOptions.Mode: ...

        @mode.setter
        def mode(self, arg: DeformableObject.InternalStrutOptions.Mode, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

    class MeshParticleOptions:
        def __init__(self) -> None: ...

        class Mode(enum.Enum):
            Surface = 0

            Filled = 1

        @property
        def mode(self) -> DeformableObject.MeshParticleOptions.Mode: ...

        @mode.setter
        def mode(self, arg: DeformableObject.MeshParticleOptions.Mode, /) -> None: ...

        @property
        def scale(self) -> float: ...

        @scale.setter
        def scale(self, arg: float, /) -> None: ...

        @property
        def spacing(self) -> float: ...

        @spacing.setter
        def spacing(self, arg: float, /) -> None: ...

        @property
        def maxFillParticles(self) -> int: ...

        @maxFillParticles.setter
        def maxFillParticles(self, arg: int, /) -> None: ...

    class MeshBuildOptions:
        def __init__(self) -> None: ...

        @property
        def particles(self) -> DeformableObject.MeshParticleOptions: ...

        @particles.setter
        def particles(self, arg: DeformableObject.MeshParticleOptions, /) -> None: ...

        @property
        def internalStruts(self) -> DeformableObject.InternalStrutOptions: ...

        @internalStruts.setter
        def internalStruts(self, arg: DeformableObject.InternalStrutOptions, /) -> None: ...

    def getNumParticles(self) -> int: ...

    def getPositions(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getVelocities(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getTriangles(self) -> list[list[int]]: ...

    def getTetrahedra(self) -> list[list[int]]: ...

    def getVisualPosition(self, index: int) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getDistanceConstraintCount(self) -> int: ...

    def getBendingConstraintCount(self) -> int: ...

    def getMaterial(self) -> DeformableObject.Material: ...

    def getCollisionBodyAt(self, index: int) -> CollisionBodyHandle: ...

    def getCollisionBodyCount(self) -> int: ...

    def setPositionOffset(self, offset: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def applyRigidTransform(self, translation: Annotated[NDArray[numpy.float64], dict(order='C')], rotation: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def pinVertex(self, index: int) -> None: ...

    def unpinVertex(self, index: int, mass: float) -> None: ...

    def addDistanceConstraint(self, i: int, j: int) -> None: ...

    def addDistanceConstraints(self, edges: Sequence[Sequence[int]]) -> None: ...

    def addInternalStruts(self, options: DeformableObject.InternalStrutOptions) -> None: ...

    def setGravity(self, gravity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

class GranularSystem(Object):
    class NormalContactModel(enum.Enum):
        LINEAR = 0

        HERTZ = 1

    class Material:
        def __init__(self) -> None: ...

        @property
        def density(self) -> float: ...

        @density.setter
        def density(self, arg: float, /) -> None: ...

        @property
        def normalStiffness(self) -> float: ...

        @normalStiffness.setter
        def normalStiffness(self, arg: float, /) -> None: ...

        @property
        def normalDamping(self) -> float: ...

        @normalDamping.setter
        def normalDamping(self, arg: float, /) -> None: ...

        @property
        def tangentialStiffness(self) -> float: ...

        @tangentialStiffness.setter
        def tangentialStiffness(self, arg: float, /) -> None: ...

        @property
        def tangentialDamping(self) -> float: ...

        @tangentialDamping.setter
        def tangentialDamping(self, arg: float, /) -> None: ...

        @property
        def friction(self) -> float: ...

        @friction.setter
        def friction(self, arg: float, /) -> None: ...

        @property
        def rollingFriction(self) -> float: ...

        @rollingFriction.setter
        def rollingFriction(self, arg: float, /) -> None: ...

        @property
        def cohesionStiffness(self) -> float: ...

        @cohesionStiffness.setter
        def cohesionStiffness(self, arg: float, /) -> None: ...

        @property
        def cohesionMaxDistance(self) -> float: ...

        @cohesionMaxDistance.setter
        def cohesionMaxDistance(self, arg: float, /) -> None: ...

        @property
        def normalContactModel(self) -> GranularSystem.NormalContactModel: ...

        @normalContactModel.setter
        def normalContactModel(self, arg: GranularSystem.NormalContactModel, /) -> None: ...

        @property
        def substeps(self) -> int: ...

        @substeps.setter
        def substeps(self, arg: int, /) -> None: ...

        @property
        def maxSpeed(self) -> float: ...

        @maxSpeed.setter
        def maxSpeed(self, arg: float, /) -> None: ...

        @property
        def maxAngularSpeed(self) -> float: ...

        @maxAngularSpeed.setter
        def maxAngularSpeed(self, arg: float, /) -> None: ...

    class ParticleMaterial:
        def __init__(self) -> None: ...

        @property
        def friction(self) -> float: ...

        @friction.setter
        def friction(self, arg: float, /) -> None: ...

        @property
        def rollingFriction(self) -> float: ...

        @rollingFriction.setter
        def rollingFriction(self, arg: float, /) -> None: ...

    class StepStats:
        @property
        def candidatePairs(self) -> int: ...

        @property
        def rigidCandidatePairs(self) -> int: ...

        @property
        def meshTriangleCandidatePairs(self) -> int: ...

        @property
        def meshInsideTriangleTests(self) -> int: ...

        @property
        def particleContacts(self) -> int: ...

        @property
        def boundaryContacts(self) -> int: ...

        @property
        def rigidContacts(self) -> int: ...

        @property
        def cohesiveContacts(self) -> int: ...

        @property
        def activeContactHistories(self) -> int: ...

        @property
        def staleContactHistoriesRemoved(self) -> int: ...

        @property
        def emittedParticles(self) -> int: ...

        @property
        def removedParticles(self) -> int: ...

        @property
        def wrappedParticles(self) -> int: ...

        @property
        def velocityClamps(self) -> int: ...

        @property
        def maxPenetration(self) -> float: ...

        @property
        def maxSpeed(self) -> float: ...

    class BoxOptions:
        def __init__(self) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def spacing(self) -> float: ...

        @spacing.setter
        def spacing(self, arg: float, /) -> None: ...

        @property
        def maxParticles(self) -> int: ...

        @maxParticles.setter
        def maxParticles(self, arg: int, /) -> None: ...

        @property
        def materialId(self) -> int: ...

        @materialId.setter
        def materialId(self, arg: int, /) -> None: ...

        @property
        def fixed(self) -> bool: ...

        @fixed.setter
        def fixed(self, arg: bool, /) -> None: ...

        @property
        def minCorner(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @minCorner.setter
        def minCorner(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

        @property
        def maxCorner(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @maxCorner.setter
        def maxCorner(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

        @property
        def velocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @velocity.setter
        def velocity(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

        @property
        def angularVelocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @angularVelocity.setter
        def angularVelocity(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    def getNumParticles(self) -> int: ...

    def getPositions(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getVelocities(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getAngularVelocities(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getRadii(self) -> list[float]: ...

    def getParticleMaterialIds(self) -> list[int]: ...

    def getFixedParticleFlags(self) -> list[int]: ...

    def getMaterial(self) -> GranularSystem.Material: ...

    def getBoundaryMaterial(self) -> GranularSystem.ParticleMaterial: ...

    def getLastStepStats(self) -> GranularSystem.StepStats: ...

    def setParticleMaterial(self, materialId: int, material: GranularSystem.ParticleMaterial) -> None: ...

    def setParticleMaterialId(self, index: int, materialId: int) -> None: ...

    def setBoundaryMaterial(self, material: GranularSystem.ParticleMaterial) -> None: ...

    def reserveParticles(self, capacity: int) -> None: ...

    def addParticle(self, position: Sequence[float], radius: float, velocity: Sequence[float] = [0.0, 0.0, 0.0], angularVelocity: Sequence[float] = [0.0, 0.0, 0.0], materialId: int = 0, fixed: bool = False) -> int: ...

    def emitBox(self, minCorner: Sequence[float], maxCorner: Sequence[float], radius: float, spacing: float, maxParticles: int, velocity: Sequence[float] = [0.0, 0.0, 0.0], angularVelocity: Sequence[float] = [0.0, 0.0, 0.0], materialId: int = 0, fixed: bool = False) -> int: ...

    def emitBoxRandom(self, minCorner: Sequence[float], maxCorner: Sequence[float], minRadius: float, maxRadius: float, spacing: float, maxParticles: int, seed: int, velocity: Sequence[float] = [0.0, 0.0, 0.0], angularVelocity: Sequence[float] = [0.0, 0.0, 0.0], materialId: int = 0, fixed: bool = False) -> int: ...

    def removeParticle(self, index: int) -> None: ...

    def removeParticlesBelowZ(self, zThreshold: float) -> int: ...

    def removeParticlesInBox(self, minCorner: Sequence[float], maxCorner: Sequence[float]) -> int: ...

    def saveGranularState(self, path: str, includeContactHistories: bool = True) -> None: ...

    def loadGranularState(self, path: str) -> None: ...

    def setVelocity(self, index: int, velocity: Sequence[float]) -> None: ...

    def setAngularVelocity(self, index: int, angularVelocity: Sequence[float]) -> None: ...

    def setParticleFixed(self, index: int, fixed: bool) -> None: ...

    def setGroundPlane(self, height: float, enabled: bool = True) -> None: ...

    def disableGroundPlane(self) -> None: ...

    def setContainerBox(self, minCorner: Sequence[float], maxCorner: Sequence[float], enabled: bool = True) -> None: ...

    def disableContainerBox(self) -> None: ...

    def setPeriodicBoundary(self, minCorner: Sequence[float], maxCorner: Sequence[float], x: bool, y: bool, z: bool) -> None: ...

    def disablePeriodicBoundary(self) -> None: ...

class DeformableTerrain(HeightMap):
    class Material:
        def __init__(self) -> None: ...

        @property
        def bekkerKc(self) -> float: ...

        @bekkerKc.setter
        def bekkerKc(self, arg: float, /) -> None: ...

        @property
        def bekkerKphi(self) -> float: ...

        @bekkerKphi.setter
        def bekkerKphi(self, arg: float, /) -> None: ...

        @property
        def sinkageExponent(self) -> float: ...

        @sinkageExponent.setter
        def sinkageExponent(self, arg: float, /) -> None: ...

        @property
        def elasticStiffness(self) -> float: ...

        @elasticStiffness.setter
        def elasticStiffness(self, arg: float, /) -> None: ...

        @property
        def damping(self) -> float: ...

        @damping.setter
        def damping(self, arg: float, /) -> None: ...

        @property
        def cohesion(self) -> float: ...

        @cohesion.setter
        def cohesion(self, arg: float, /) -> None: ...

        @property
        def frictionCoefficient(self) -> float: ...

        @frictionCoefficient.setter
        def frictionCoefficient(self, arg: float, /) -> None: ...

        @property
        def shearDisplacement(self) -> float: ...

        @shearDisplacement.setter
        def shearDisplacement(self, arg: float, /) -> None: ...

    class State:
        def __init__(self) -> None: ...

        @property
        def sinkage(self) -> list[float]: ...

        @sinkage.setter
        def sinkage(self, arg: Sequence[float], /) -> None: ...

        @property
        def plasticSinkage(self) -> list[float]: ...

        @plasticSinkage.setter
        def plasticSinkage(self, arg: Sequence[float], /) -> None: ...

        @property
        def shear(self) -> list[float]: ...

        @shear.setter
        def shear(self, arg: Sequence[float], /) -> None: ...

    class StepStats:
        @property
        def candidateCells(self) -> int: ...

        @property
        def rayQueries(self) -> int: ...

        @property
        def contacts(self) -> int: ...

        @property
        def updatedVertices(self) -> int: ...

        @property
        def contactArea(self) -> float: ...

        @property
        def normalForce(self) -> float: ...

    def getSoilMaterial(self) -> DeformableTerrain.Material: ...

    def getUndeformedHeights(self) -> list[float]: ...

    def getSoilState(self) -> DeformableTerrain.State: ...

    def getSoilStepStats(self) -> DeformableTerrain.StepStats: ...

    def setSoilState(self, state: DeformableTerrain.State) -> None: ...

    def resetSoil(self) -> None: ...

    def pressure(self, sinkage: float, plasticSinkage: float, contactWidth: float) -> float: ...

class Tendon:
    class Type(enum.Enum):
        Spatial = 0

        Fixed = 1

    class Site:
        def __init__(self, object: Object | None = None, localIndex: int = 0, position: Sequence[float] = [0.0, 0.0, 0.0]) -> None: ...

        @property
        def object(self) -> Object: ...

        @object.setter
        def object(self, arg: Object, /) -> None: ...

        @property
        def localIndex(self) -> int: ...

        @localIndex.setter
        def localIndex(self, arg: int, /) -> None: ...

        @property
        def position(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @position.setter
        def position(self, arg: Sequence[float], /) -> None: ...

    class PathElement:
        def __init__(self) -> None: ...

        class Kind(enum.Enum):
            Site = 0

            Sphere = 1

            Cylinder = 2

            Pulley = 3

        @staticmethod
        def via(site: Tendon.Site) -> Tendon.PathElement: ...

        @staticmethod
        def sphere(center: Tendon.Site, radius: float) -> Tendon.PathElement: ...

        @staticmethod
        def cylinder(center: Tendon.Site, radius: float, axis: Sequence[float] = [0.0, 0.0, 1.0]) -> Tendon.PathElement: ...

        @staticmethod
        def pulley(divisor: float) -> Tendon.PathElement: ...

        def withSideSite(self, side: Tendon.Site) -> Tendon.PathElement: ...

        @property
        def kind(self) -> Tendon.PathElement.Kind: ...

        @kind.setter
        def kind(self, arg: Tendon.PathElement.Kind, /) -> None: ...

        @property
        def site(self) -> Tendon.Site: ...

        @site.setter
        def site(self, arg: Tendon.Site, /) -> None: ...

        @property
        def radius(self) -> float: ...

        @radius.setter
        def radius(self, arg: float, /) -> None: ...

        @property
        def hasSideSite(self) -> bool: ...

        @hasSideSite.setter
        def hasSideSite(self, arg: bool, /) -> None: ...

        @property
        def sideSite(self) -> Tendon.Site: ...

        @sideSite.setter
        def sideSite(self, arg: Tendon.Site, /) -> None: ...

        @property
        def divisor(self) -> float: ...

        @divisor.setter
        def divisor(self, arg: float, /) -> None: ...

        @property
        def axis(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @axis.setter
        def axis(self, arg: Sequence[float], /) -> None: ...

    class JointTerm:
        def __init__(self, system: ArticulatedSystem, joint: str, coefficient: float = 1.0) -> None: ...

        @property
        def system(self) -> ArticulatedSystem: ...

        @system.setter
        def system(self, arg: ArticulatedSystem, /) -> None: ...

        @property
        def joint(self) -> str: ...

        @joint.setter
        def joint(self, arg: str, /) -> None: ...

        @property
        def coefficient(self) -> float: ...

        @coefficient.setter
        def coefficient(self, arg: float, /) -> None: ...

    class Properties:
        def __init__(self) -> None: ...

        @property
        def stiffness(self) -> float: ...

        @stiffness.setter
        def stiffness(self, arg: float, /) -> None: ...

        @property
        def damping(self) -> float: ...

        @damping.setter
        def damping(self, arg: float, /) -> None: ...

        @property
        def springLower(self) -> float: ...

        @springLower.setter
        def springLower(self, arg: float, /) -> None: ...

        @property
        def springUpper(self) -> float: ...

        @springUpper.setter
        def springUpper(self, arg: float, /) -> None: ...

        @property
        def lowerLimit(self) -> float: ...

        @lowerLimit.setter
        def lowerLimit(self, arg: float, /) -> None: ...

        @property
        def upperLimit(self) -> float: ...

        @upperLimit.setter
        def upperLimit(self, arg: float, /) -> None: ...

        @property
        def limitMargin(self) -> float: ...

        @limitMargin.setter
        def limitMargin(self, arg: float, /) -> None: ...

        @property
        def limitCompliance(self) -> float: ...

        @limitCompliance.setter
        def limitCompliance(self, arg: float, /) -> None: ...

        @property
        def frictionLoss(self) -> float: ...

        @frictionLoss.setter
        def frictionLoss(self, arg: float, /) -> None: ...

        @property
        def frictionCompliance(self) -> float: ...

        @frictionCompliance.setter
        def frictionCompliance(self, arg: float, /) -> None: ...

        @property
        def armature(self) -> float: ...

        @armature.setter
        def armature(self, arg: float, /) -> None: ...

        @property
        def positionCorrection(self) -> float: ...

        @positionCorrection.setter
        def positionCorrection(self, arg: float, /) -> None: ...

        @property
        def actuationLower(self) -> float: ...

        @actuationLower.setter
        def actuationLower(self, arg: float, /) -> None: ...

        @property
        def actuationUpper(self) -> float: ...

        @actuationUpper.setter
        def actuationUpper(self, arg: float, /) -> None: ...

        @property
        def width(self) -> float: ...

        @width.setter
        def width(self, arg: float, /) -> None: ...

        @property
        def color(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @color.setter
        def color(self, arg: Sequence[float], /) -> None: ...

    class Drive:
        def __init__(self) -> None: ...

        @property
        def force(self) -> float: ...

        @force.setter
        def force(self, arg: float, /) -> None: ...

        @property
        def targetLength(self) -> float: ...

        @targetLength.setter
        def targetLength(self, arg: float, /) -> None: ...

        @property
        def targetVelocity(self) -> float: ...

        @targetVelocity.setter
        def targetVelocity(self, arg: float, /) -> None: ...

        @property
        def positionGain(self) -> float: ...

        @positionGain.setter
        def positionGain(self, arg: float, /) -> None: ...

        @property
        def velocityGain(self) -> float: ...

        @velocityGain.setter
        def velocityGain(self, arg: float, /) -> None: ...

        @property
        def activationTime(self) -> float: ...

        @activationTime.setter
        def activationTime(self, arg: float, /) -> None: ...

    class VisualSegment:
        @property
        def start(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @property
        def end(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @property
        def wrapped(self) -> bool: ...

    def getType(self) -> Tendon.Type: ...

    def getName(self) -> str: ...

    def getLength(self) -> float: ...

    def getVelocity(self) -> float: ...

    def getReferenceLength(self) -> float: ...

    def getForce(self) -> float: ...

    def getTension(self) -> float: ...

    def getActuationForce(self) -> float: ...

    def getLimitForce(self) -> float: ...

    def getFrictionForce(self) -> float: ...

    def getPotentialEnergy(self) -> float: ...

    def getKineticEnergy(self) -> float: ...

    def isEnabled(self) -> bool: ...

    def setName(self, name: str) -> None: ...

    def setProperties(self, properties: Tendon.Properties) -> None: ...

    def setDrive(self, drive: Tendon.Drive) -> None: ...

    def setTension(self, tension: float) -> None: ...

    def setActuationForce(self, force: float) -> None: ...

    def setEnabled(self, enabled: bool) -> None: ...

    def getProperties(self) -> Tendon.Properties: ...

    def getDrive(self) -> Tendon.Drive: ...

    def getPath(self) -> list[Tendon.PathElement]: ...

    def getJoints(self) -> list[Tendon.JointTerm]: ...

    def getVisualSegments(self) -> list[Tendon.VisualSegment]: ...

    def updateGeometry(self, visuals: bool = False) -> None: ...

class TendonCoupling:
    class Properties:
        def __init__(self) -> None: ...

        @property
        def coefficients(self) -> list[float]: ...

        @coefficients.setter
        def coefficients(self, arg: Sequence[float], /) -> None: ...

        @property
        def compliance(self) -> float: ...

        @compliance.setter
        def compliance(self, arg: float, /) -> None: ...

        @property
        def positionCorrection(self) -> float: ...

        @positionCorrection.setter
        def positionCorrection(self, arg: float, /) -> None: ...

        @property
        def enabled(self) -> bool: ...

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

    def getName(self) -> str: ...

    def getFirst(self) -> Tendon: ...

    def getSecond(self) -> Tendon: ...

    def getProperties(self) -> TendonCoupling.Properties: ...

    def setProperties(self, properties: TendonCoupling.Properties) -> None: ...

    def getForce(self) -> float: ...

class WorldParameterContainer:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def param(self) -> str: ...

    @param.setter
    def param(self, arg: str, /) -> None: ...

class Polyline:
    def setColor(self, r: float, g: float, b: float, a: float) -> None:
        """
        Set the color of the visual body.
                Args:
                    r: red value (1 is max).
                    g: green value (1 is max).
                    b: blue value (1 is max).
                    a: alpha value (1 is max).
        """

    @property
    def color(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @color.setter
    def color(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def points(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @points.setter
    def points(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def width(self) -> float: ...

    @width.setter
    def width(self, arg: float, /) -> None: ...

    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

    def clearPoints(self) -> None:
        """Clear points."""

    def addPoint(self, point: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]) -> None:
        """
        Add a point.
                Args:
                  point: 3d numpy array
        """

class ArticulatedSystemVisual:
    def setColor(self, r: float, g: float, b: float, a: float) -> None:
        """
        Set the color of the visual body.
                Args:
                    r: red value (1 is max).
                    g: green value (1 is max).
                    b: blue value (1 is max).
                    a: alpha value (1 is max).
        """

    @property
    def color(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @color.setter
    def color(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def obj(self) -> ArticulatedSystem: ...

    def setGeneralizedCoordinate(self, gc: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Set the generalized Coordinate.
                Args:
                    gc: the generalizedCoordinate.
        """

    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

class Visual:
    @overload
    def setPosition(self, position: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]) -> None:
        """
        Set the position of the visual body.
                Args:
                    position: the new position of the visual body.
        """

    @overload
    def setPosition(self, x: float, y: float, z: float) -> None: ...

    @overload
    def setOrientation(self, orientation: Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')]) -> None:
        """
        Set the orientation of the visual body.
                Args:
                    orientation: the new orientation of the visual body.
        """

    @overload
    def setOrientation(self, w: float, x: float, y: float, z: float) -> None: ...

    def setColor(self, r: float, g: float, b: float, a: float) -> None:
        """
        Set the color of the visual body.
                Args:
                    r: red value (1 is max).
                    g: green value (1 is max).
                    b: blue value (1 is max).
                    a: alpha value (1 is max).
        """

    @property
    def color(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @color.setter
    def color(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def size(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @size.setter
    def size(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def material(self) -> str: ...

    @material.setter
    def material(self, arg: str, /) -> None: ...

    @property
    def glow(self) -> bool: ...

    @glow.setter
    def glow(self, arg: bool, /) -> None: ...

    @property
    def shadow(self) -> bool: ...

    @shadow.setter
    def shadow(self, arg: bool, /) -> None: ...

    @property
    def type(self) -> ShapeType: ...

    @type.setter
    def type(self, arg: ShapeType, /) -> None: ...

    def setBoxSize(self, x: float, y: float, z: float) -> None:
        """
        Set the size of the visual box.
                Args:
                    x: length (in x axis).
                    y: width (in y axis).
                    z: height (in z axis).
        """

    def setCylinderSize(self, radius: float, height: float) -> None:
        """
        Set the size of the visual cylinder.
                Args:
                    radius: radius.
                    height: height.
        """

    def setCapsuleSize(self, radius: float, height: float) -> None:
        """
        Set the size of the visual capsule.
                Args:
                    radius: radius.
                    height: height.
        """

    def setSphereSize(self, radius: float) -> None:
        """
        Set the size of the visual sphere.
                Args:
                    x: radius.
        """

    def getPosition(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def getOrientation(self) -> Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')]: ...

    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

class VisualMesh(Visual):
    def updateMesh(self, vertexArray: Sequence[float], colorArray: Sequence[int]) -> None:
        """
        Set new mesh vertices and colors
          Args:
            vertex array: new vertex array
            color array: new color array
        """

    def isUpdated(self) -> bool: ...

class HeightMapVisual:
    @overload
    def setColor(self, r: float, g: float, b: float, a: float) -> None: ...

    @overload
    def setColor(self, color: Sequence[ColorRGB]) -> None: ...

    @property
    def color(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @color.setter
    def color(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def obj(self) -> HeightMap: ...

    def update(self, centerX: float, centerY: float, sizeX: float, sizeY: float, height: Sequence[float]) -> None: ...

    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

class InstancedVisuals:
    def count(self) -> int: ...

    def clearInstances(self) -> None: ...

    def resize(self, numberOfObjects: int) -> None: ...

    @overload
    def addInstance(self, position: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], orientation: Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')], scale: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], colorWeight: float = 0.0) -> None: ...

    @overload
    def addInstance(self, position: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], orientation: Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')], colorWeight: float = 0.0) -> None: ...

    @overload
    def addInstance(self, position: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], colorWeight: float = 0.0) -> None: ...

    def removeInstance(self, id: int) -> None: ...

    def setPosition(self, id: int, position: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]) -> None: ...

    def setOrientation(self, id: int, orientation: Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')]) -> None: ...

    def setScale(self, id: int, scale: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]) -> None: ...

    def setColor(self, color1: Annotated[NDArray[numpy.float64], dict(order='C')], color2: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

    def setColorWeight(self, id: int, weight: float) -> None: ...

    def getPosition(self, id: int) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def getOrientation(self, id: int) -> Annotated[NDArray[numpy.float64], dict(shape=(4), order='C')]: ...

    def getScale(self, id: int) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

class PointCloud:
    def resize(self, size: int) -> None: ...

    @property
    def pointSize(self) -> float: ...

    @pointSize.setter
    def pointSize(self, arg: float, /) -> None: ...

    @property
    def position(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @position.setter
    def position(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def color(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @color.setter
    def color(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

class Chart:
    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

    def getType(self) -> int: ...

class TimeSeriesGraph(Chart):
    def addDataPoints(self, time: float, data: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None: ...

class BarChart(Chart):
    def setData(self, data: Sequence[float]) -> None: ...

class RaisimServer:
    def __init__(self, arg: World, /) -> None: ...

    def setupSocket(self, port: int = 8080) -> None: ...

    def acceptConnection(self, microseconds: int) -> None: ...

    def closeConnection(self) -> None: ...

    def waitForNewClients(self, microseconds: int) -> bool: ...

    def waitForMessageFromClient(self, seconds: int) -> bool: ...

    def waitForMessageToClient(self, seconds: int) -> bool: ...

    def launchServer(self, port: int = 8080) -> None: ...

    def getPort(self) -> int: ...

    def setBindLoopbackOnly(self, loopbackOnly: bool) -> None: ...

    def setSceneFileSharing(self, enabled: bool) -> None: ...

    def pauseSimulation(self) -> None: ...

    def resumeSimulation(self) -> None: ...

    def isSimulationPaused(self) -> bool: ...

    def stepSimulation(self, n: int = 1) -> None: ...

    def killServer(self) -> None: ...

    def integrateWorldThreadSafe(self) -> None: ...

    def applyInteractionForce(self) -> None: ...

    def lockVisualizationServerMutex(self) -> None: ...

    def unlockVisualizationServerMutex(self) -> None: ...

    def hibernate(self) -> None: ...

    def wakeup(self) -> None: ...

    def isConnected(self) -> bool: ...

    def isTerminateRequested(self) -> bool: ...

    def needsSensorUpdate(self) -> bool: ...

    def processRequests(self) -> bool: ...

    def requestSaveScreenshot(self) -> None: ...

    def startRecordingVideo(self, videoName: str) -> None:
        """
        Start recording RaisimUnity visualization. RaisimUnity only supports video recording in linux.
        Args:
          videoName: name of the video file
        """

    def stopRecordingVideo(self) -> None: ...

    def addSlider(self, name: str, min: float, max: float, value: float) -> int:
        """
        Add a slider to the Controls window of the rayrai TCP viewer: a value the viewer's user sets within
        [min, max] while the application reads it with getSlider().

        Returns:
            Its index for getSlider().
        """

    def getSlider(self, index: int) -> float:
        """The value of a slider (addSlider()), as the viewer's user last set it."""

    def setCameraPositionAndLookAt(self, position: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')], lookAt: Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]) -> None: ...

    def focusOn(self, object: Object) -> None:
        """
        Focus camera on an object
                    Args:
                      object: name
        """

    def addVisualArticulatedSystem(self, name: str, urdfFile: str, colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = 0) -> ArticulatedSystemVisual:
        """
        Add a visual capsule without physics
        Args:
          name: name
          urdfFile: radius
          colorR: red value (max:1),
          colorG: green value (max:1),
          colorB: blue value (max:1),
          colorA: alpha value (max:1)

        Returns:
            pointer to the visual articulated system
        """

    def removeVisualArticulatedSystem(self, visual: ArticulatedSystemVisual) -> None: ...

    def getVisualArticulatedSystem(self, name: str) -> ArticulatedSystemVisual: ...

    def addInstancedVisuals(self, name: str, type: ShapeType, size: Annotated[NDArray[numpy.float64], dict(order='C')], color1: Annotated[NDArray[numpy.float64], dict(order='C')], color2: Annotated[NDArray[numpy.float64], dict(order='C')]) -> InstancedVisuals: ...

    def addVisualCapsule(self, name: str, radius: float, length: float, colorR: float = 1, colorG: float = 1, colorB: float = 1, colorA: float = 1, material: str = '', glow: bool = False, shadow: bool = False) -> Visual:
        """
        Add a visual capsule without physics
        Args:
          name: name
          radius: radius
          length: length
          colorR: red value (max:1),
          colorG: green value (max:1),
          colorB: blue value (max:1),
          colorA: alpha value (max:1),
          material = "",
          glow(not supported) = false
          shadow(not supported) = false

        Returns:
            pointer to the visual capsule
        """

    def addVisualCylinder(self, name: str, radius: float, length: float, colorR: float = 1, colorG: float = 1, colorB: float = 1, colorA: float = 1, material: str = '', glow: bool = False, shadow: bool = False) -> Visual:
        """
        Add a visual cylinder without physics
                Args:
                    name: name
                    radius: radius
                    length: length
                    colorR: red value (max:1),
                    colorG: green value (max:1),
                    colorB: blue value (max:1),
                    colorA: alpha value (max:1),
                    material = "",
                    glow(not supported) = false
                    shadow(not supported) = false

                Returns:
                    pointer to the visual cylinder
        """

    def addVisualArrow(self, name: str, radius: float, height: float, colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = -1, glow: bool = False, shadow: bool = False) -> Visual:
        """
        Add a visual cylinder without physics
                Args:
                    name: name
                    radius: radius
                    height: length
                    colorR: red value (max:1),
                    colorG: green value (max:1),
                    colorB: blue value (max:1),
                    colorA: alpha value (max:1),
                    glow(not supported) = false
                    shadow(not supported) = false

                Returns:
                    pointer to the visual cylinder
        """

    def addVisualBox(self, name: str, x: float, y: float, z: float, colorR: float = 1, colorG: float = 1, colorB: float = 1, colorA: float = 1, material: str = '', glow: bool = False, shadow: bool = False) -> Visual:
        """
        Add a visual box without physics
                Args:
                    name: name
                    x: length (in x axis)
                    y: width (in y axis)
                    z: height (in z axis)
                    colorR: red value (max:1),
                    colorG: green value (max:1),
                    colorB: blue value (max:1),
                    colorA: alpha value (max:1),
                    material = "",
                    glow(not supported) = false
                    shadow(not supported) = false

                Returns:
                    pointer to the visual box
        """

    def addVisualSphere(self, name: str, radius: float, colorR: float = 1, colorG: float = 1, colorB: float = 1, colorA: float = 1, material: str = '', glow: bool = False, shadow: bool = False) -> Visual:
        """
        Add a visual sphere without physics
                Args:
                    name: name
                    radius: radius
                    colorR: red value (max:1),
                    colorG: green value (max:1),
                    colorB: blue value (max:1),
                    colorA: alpha value (max:1),
                    material = "",
                    glow(not supported) = false
                    shadow(not supported) = false
                Returns:
                    pointer to the visual sphere
        """

    def addVisualPolyLine(self, name: str) -> Polyline:
        """
        Add a visual polyline without physics
                Args:
                    name: name

                Returns:
                    pointer to the visual polyline
        """

    def getVisualPolyLine(self, name: str) -> Polyline: ...

    def removeVisualPolyLine(self, name: str) -> None: ...

    def addPointCloud(self, name: str) -> PointCloud: ...

    def removePointCloud(self, name: str) -> None: ...

    @overload
    def addVisualHeightMap(self, name: str, pngFileName: str, centerX: float, centerY: float, xScale: float, yScale: float, heightScale: float, heightOffset: float, colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = -1) -> HeightMapVisual: ...

    @overload
    def addVisualHeightMap(self, name: str, fileName: str, centerX: float, centerY: float, colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = -1) -> HeightMapVisual: ...

    @overload
    def addVisualHeightMap(self, name: str, centerX: float, centerY: float, terrainProperties: TerrainProperties, colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = -1) -> HeightMapVisual: ...

    @overload
    def addVisualHeightMap(self, name: str, xSamples: int, ySamples: int, xSize: float, ySize: float, centerX: float, centerY: float, height: Sequence[float], colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = -1) -> HeightMapVisual: ...

    @overload
    def addVisualHeightMap(self, name: str, heightMap: HeightMap, colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = -1) -> HeightMapVisual: ...

    def removeVisualHeightMap(self, heightMap: HeightMapVisual) -> None: ...

    def getVisualHeightMap(self, name: str) -> HeightMapVisual: ...

    def getVisualObject(self, name: str) -> Visual: ...

    def removeVisualObject(self, name: str) -> None: ...

    @overload
    def addVisualMesh(self, name: str, file: str, scale: float = 1, colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = -1, glow: bool = False, shadow: bool = False) -> Visual:
        """
        Add a visual mesh without physics
          Args:
              name: name of the visual object
              file: location of the mesh file
              scale: scale of the visual mesh
              colorR: red color value
              colorG: green color value
              colorB: blue color value
              colorA: alpha color value
              glow(not supported): if glow
              shadow(not supported): if casts shadow

          Returns:
              pointer to the visual mesh
        """

    @overload
    def addVisualMesh(self, name: str, file: str, scale: Annotated[NDArray[numpy.float64], dict(order='C')], colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = -1, glow: bool = False, shadow: bool = False) -> Visual:
        """
        Add a visual mesh without physics
          Args:
              name: name of the visual object
              file: location of the mesh file
              scale: scale of the visual mesh (vec3)
              colorR: red color value
              colorG: green color value
              colorB: blue color value
              colorA: alpha color value
              glow(not supported): if glow
              shadow(not supported): if casts shadow

          Returns:
              pointer to the visual mesh
        """

    @overload
    def addVisualMesh(self, name: str, vertex: Sequence[float], color: Sequence[int], index: Sequence[int], colorR: float = 0, colorG: float = 0, colorB: float = 0, colorA: float = 1, glow: bool = False, shadow: bool = False) -> VisualMesh:
        """
        Add a visual mesh without physics
          Args:
              name: name of the visual object
              vertex: vertex array which contains the positions of the vertices
              color: color array which contains the colors of the vertices
              index: index array which contains the vertex index triplets that form a triangle
              colorR: red color value
              colorG: green color value
              colorB: blue color value
              colorA: alpha color value
              glow(not supported): if glow
              shadow(not supported): if casts shadow

          Returns:
              pointer to the visual mesh
        """

    def addTimeSeriesGraph(self, title: str, names: Sequence[str], xAxis: str, yAxis: str) -> TimeSeriesGraph: ...

    def addBarChart(self, title: str, names: Sequence[str]) -> BarChart: ...

class World:
    """Raisim world."""

    @overload
    def __init__(self) -> None:
        """Initialize the World."""

    @overload
    def __init__(self, configFile: str) -> None:
        """Initialize the World from the given config file."""

    @overload
    def __init__(self, configFile: str, params: Sequence[WorldParameterContainer] = []) -> None:
        """
        Initialize the World from the given config file with parameter substitution.
        """

    @staticmethod
    def setLicenseFile(licenseFile: str) -> None:
        """
        Set the license file path.

        Args:
            path: Path to the license file.
        """

    @staticmethod
    def setActivationKey(activationKey: str) -> None:
        """
        Set the activation key file path.

        Args:
            path: Path to the license file.
        """

    @overload
    def exportToXml(self, dir: str, fileName: str) -> None:
        """
        Export the world to an XML file.

        Args:
            dir (str): output directory.
            fileName (str): output filename.
        """

    @overload
    def exportToXml(self, path: str) -> None:
        """
        Export the world to an XML file.

        Args:
            path (str): output file path.
        """

    def getConfigFile(self) -> str:
        """Get the configuration file path used to initialize the world."""

    def setTimeStep(self, dt: float) -> None:
        """
        Set the given time step `dt` in the simulator.

        Args:
            dt (float): time step to be set in the simulator.
        """

    def getTimeStep(self) -> float:
        """
        Get the current time step that has been set in the simulator.

        Returns:
            float: time step.
        """

    def addSphere(self, radius: float, mass: float, material: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> Sphere:
        """
        Add dynamically a sphere into the world.

        Args:
            radius (float): radius of the sphere.
            mass (float): mass of the sphere.
            material (str): material to be applied to the sphere.
            collision_group (unsigned long): collision group.
            collision_mask (unsigned long): collision mask.

        Returns:
            Sphere: the sphere instance.
        """

    def addBox(self, x: float, y: float, z: float, mass: float, material: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> Box:
        """
        Add dynamically a box into the world.

        Args:
            x (float): length along the x axis.
            y (float): length along the y axis.
            z (float): length along the z axis.
            mass (float): mass of the box.
            material (str): material to be applied to the box.
            collision_group (unsigned long): collision group.
            collision_mask (unsigned long): collision mask.

        Returns:
            Box: the box instance.
        """

    def addCylinder(self, radius: float, height: float, mass: float, material: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> Cylinder:
        """
        Add dynamically a cylinder into the world.

        Args:
            radius (float): radius of the cylinder.
            height (float): height of the cylinder.
            mass (float): mass of the cylinder.
            material (str): material to be applied to the cylinder.
            collision_group (unsigned long): collision group.
            collision_mask (unsigned long): collision mask.

        Returns:
            Cylinder: the cylinder instance.
        """

    def addCapsule(self, radius: float, height: float, mass: float, material: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> Capsule:
        """
        Add dynamically a capsule into the world.

        Args:
            radius (float): radius of the capsule.
            height (float): height of the capsule.
            mass (float): mass of the capsule.
            material (str): material to be applied to the capsule.
            collision_group (unsigned long): collision group.
            collision_mask (unsigned long): collision mask.

        Returns:
            Capsule: the capsule instance.
        """

    def addGround(self, height: float = 0.0, material: str = 'default', collision_mask: int = 18446744073709551615) -> Ground:
        """
        Add dynamically a ground into the world.

        Args:
            height (float): height of the ground.
            material (str): material to be applied to the ground.
            collision_mask (unsigned long): collision mask.

        Returns:
            Ground: the ground instance.
        """

    @overload
    def addHeightMap(self, x_samples: int, y_samples: int, x_scale: float, y_scale: float, x_center: float, y_center: float, heights: Sequence[float], material: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> HeightMap:
        """
        Add a heightmap into the world.

        	    Args:
        	        x_samples (int): the number of samples in x.
        	        y_samples (int): the number of samples in y.
                    x_scale (float): the scale in the x direction.
                    y_scale (float): the scale in the y direction.
                    x_center (float): the x center of the heightmap in the world.
                    y_center (float): the y center of the heightmap in the world.
                    heights (list[float]): list of desired heights.
                    material (str): material.
                    group (CollisionGroup): collision group.
                    mask (CollisionGroup): collision mask.

        	    Returns:
        	        HeightMap: the heightmap instance.
        """

    @overload
    def addHeightMap(self, filename: str, x_center: float, y_center: float, material: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> HeightMap:
        """
        Add a heightmap into the world.

        	    Args:
        	        filename (str): raisim heightmap filename.
                    x_center (float): the x center of the heightmap in the world.
                    y_center (float): the y center of the heightmap in the world.
                    material (str): material.
                    group (CollisionGroup): collision group.
                    mask (CollisionGroup): collision mask.

        	    Returns:
        	        HeightMap: the heightmap instance.
        """

    @overload
    def addHeightMap(self, filename: str, x_center: float, y_center: float, x_size: float, y_size: float, height_scale: float, height_offset: float, material: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> HeightMap:
        """
        Add a heightmap into the world.

        	    Args:
        	        filename (str): filename to the PNG.
        	        x_center (float): the x center of the heightmap in the world.
                    y_center (float): the y center of the heightmap in the world.
                    x_size (float): the size in the x direction.
                    y_size (float): the size in the y direction.
        	        height_scale (float): the height scale.
        	        height_offset (float): the height offset.
                    material (str): material.
                    group (CollisionGroup): collision group.
                    mask (CollisionGroup): collision mask.

        	    Returns:
        	        HeightMap: the heightmap instance.
        """

    @overload
    def addHeightMap(self, x_center: float, y_center: float, terrain_properties: TerrainProperties, material: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> HeightMap:
        """
        Add a heightmap into the world.

        	    Args:
                    x_center (float): the x center of the heightmap in the world.
                    y_center (float): the y center of the heightmap in the world.
                    terrain_properties (TerrainProperties): the terrain properties.
                    material (str): material.
                    group (CollisionGroup): collision group.
                    mask (CollisionGroup): collision mask.

        	    Returns:
        	        HeightMap: the heightmap instance.
        """

    @overload
    def addArticulatedSystem(self, urdf_path: str, res_path: str = '', joint_order: Sequence[str] = [], collision_group: int = 1, collision_mask: int = 18446744073709551615, options: ArticulatedSystemOption = ...) -> ArticulatedSystem:
        r"""
        Add an articulated system in the world.

        	    Args:
                    urdf_path (str): path to the URDF file.
                    res_path (str): path to the resource directory. Leave it empty (\'\') if it is the urdf file directory.
                    joint_order (list[str]): joint order.
                    group (CollisionGroup): collision group.
                    mask (CollisionGroup): collision mask.
                    options (ArticulatedSystemOption): articulated system options.

        	    Returns:
        	        ArticulatedSystem: the articulated system instance.
        """

    @overload
    def addArticulatedSystem(self, filePathOrURDFScript: str, modules: Sequence[str], resPath: str = '', jointOrder: Sequence[str] = [], collision_group: int = 1, collision_mask: int = 18446744073709551615, options: ArticulatedSystemOption = ...) -> ArticulatedSystem: ...

    def addCompound(self, children: Sequence[Compound.CompoundObjectChild], mass: float, center_of_mass: Annotated[NDArray[numpy.float64], dict(order='C')], inertia: Annotated[NDArray[numpy.float64], dict(order='C')], collision_group: int = 1, collision_mask: int = 18446744073709551615) -> Compound:
        """
        Add a compound body in the world.

        	    Args:
                    children (list[CompoundObjectChild]): list of child object instance.
                    mass (float): mass of the compound object.
                    center_of_mass(np.array[float[3]]): center of mass of the compound
                    inertia (np.array[float[3,3]]): inertia matrix of the object.
                    group (CollisionGroup): collision group.
                    mask (CollisionGroup): collision mask.

        	    Returns:
        	        Compound: the compound body instance.
        """

    def addMesh(self, file_name: str, mass: float, inertia: Annotated[NDArray[numpy.float64], dict(order='C')], com: Annotated[NDArray[numpy.float64], dict(order='C')], scale: float = 1.0, material: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615, collision_mode: MeshCollisionMode = MeshCollisionMode.CONVEXIFY) -> Mesh:
        """
        Add a mesh in the world.

        	    Args:
                    file_name (str): full path of the mesh file.
                    mass (float): mass of the compound object.
                    inertia (np.array[float[3,3]]): inertia matrix of the object.
                    com (np.array[float[3,1]]): the location of the center of mass
                    material (str): material
                    group (CollisionGroup): collision group.
                    mask (CollisionGroup): collision mask.
                    collision_mode (MeshCollisionMode): collision representation.

        	    Returns:
        	        Mesh: the mesh instance.
        """

    def addSpatialTendon(self, name: str, path: Sequence[Tendon.PathElement], properties: Tendon.Properties = ...) -> Tendon: ...

    def addFixedTendon(self, name: str, joints: Sequence[Tendon.JointTerm], properties: Tendon.Properties = ...) -> Tendon: ...

    def addTendonCoupling(self, name: str, first: Tendon, second: Tendon | None = None, properties: TendonCoupling.Properties = ...) -> TendonCoupling: ...

    def getTendon(self, name: str) -> Tendon: ...

    def removeTendon(self, tendon: Tendon) -> None: ...

    def removeTendonCoupling(self, coupling: TendonCoupling) -> None: ...

    @overload
    def getObject(self, name: str) -> Object:
        """
        Get the specified object instance from its unique name.

        	    Args:
                    name (str): unique name of the object instance we want to get.

        	    Returns:
        	        Object, None: the specified object instance. None, if it didn't find the object.
        """

    @overload
    def getObject(self, world_index: int) -> Object: ...

    def getConfigurationNumber(self) -> int:
        """
        Get the number of elements that are in the world. The returned number is updated everytime that we add or
        remove an object from the world.

        Returns:
            int: the number of objects in the world.
        """

    def removeObject(self, obj: Object) -> None:
        """
        Remove dynamically an object from the world.

        Args:
            obj (Object): the object to be removed from the world.
        """

    def integrate(self) -> None:
        """
        this function is simply calling both `integrate1()` and `integrate2()` one-by-one.
        """

    def integrate1(self) -> None:
        """
        It performs:
        1. deletion contacts from previous time step
        2. collision detection
        3. register contacts to each body
        4. calls `preContactSolverUpdate1()` of each object
        """

    def discardPendingStep(self) -> None:
        """Cancel preparation without advancing time; clear contacts and allow scene edits.

        Does nothing without a pending step. Does not undo integrate1() preparation side effects.
        """

    def integrate2(self) -> None:
        """
        It performs
        1. calls `preContactSolverUpdate2()` of each body
        2. run collision solver
        3. calls `integrate` method of each object
        """

    def integrateNoContactDetection(self) -> None:
        """
        It performs
        1. deletion contacts from previous time step
        2. updates collision objects
        3. calls `preContactSolverUpdate1()` of each object
        It skips contact detection and contact problem construction.
        """

    def getContactProblem(self) -> list[contact.Single3DContactProblem]:
        """Return the list of contacts."""

    def getObjList(self) -> list[Object]:
        """
        Return the list of object instances that are in the world.

        Returns:
            list[Object]: list of object instances.
        """

    def updateMaterialProp(self, prop: MaterialManager) -> None:
        """
        Update material property.

        Args:
            prop (MaterialManager): material manager property instance.
        """

    def getMaterialPairProperties(self, material1: str, material2: str) -> MaterialPairProperties: ...

    @overload
    def setMaterialPairProp(self, material1: str, material2: str, friction: float, restitution: float, threshold: float) -> None:
        """
        Set material pair properties.

        Args:
            material1 (str): first material.
            material2 (str): second material.
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            threshold (float): restitution threshold.
        """

    @overload
    def setMaterialPairProp(self, material1: str, material2: str, friction: float, restitution: float, restitution_threshold: float, static_friction: float, static_friction_velocity_threshold: float) -> None:
        """
        Set material pair properties.

        Args:
            material1 (str): first material.
            material2 (str): second material.
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            restitution threshold (float): restitution threshold.
            static friction (float): coefficient of static friction
            static friction velocity threshold (float): If the relative velocity of two contact points is bigger than this value, then the dynamic coefficient of friction is applied. Otherwise, the coefficient of friction is interpolated between the static and dynamic one proportional to the relative velocity.
        """

    @overload
    def setDefaultMaterial(self, friction: float, restitution: float, threshold: float, static_friction: float, static_friction_velocity_threshold: float) -> None:
        """
        Set the default material.

        Args:
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            restitution threshold (float): restitution threshold.
            static friction (float): coefficient of static friction.
            static friction velocity threshold (float): If the relative velocity of two contact points is bigger than this value, then the dynamic coefficient of friction is applied. Otherwise, the coefficient of friction is interpolated between the static and dynamic one proportional to the relative velocity.
        """

    @overload
    def setDefaultMaterial(self, friction: float, restitution: float, restitution_threshold: float) -> None:
        """
        Set the default material.

        Args:
            friction (float): coefficient of friction.
            restitution (float): coefficient of restitution.
            restitution threshold (float): restitution threshold.
        """

    def getGravity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the gravity vector from the world.

        Returns:
            np.array[float[3]]: gravity vector.
        """

    def setGravity(self, gravity: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Set the gravity vector in the world.

        Args:
            np.array[float[3]]: gravity vector.
        """

    def setAmbientFluid(self, fluid: FluidMedium) -> None:
        """
        Fill the world with a fluid, such as air (FluidMedium.air()), outside the volumes of liquid.

        Args:
            fluid (FluidMedium): the fluid; a density of zero removes it.
        """

    def getAmbientFluid(self) -> FluidMedium:
        """Get the fluid outside the volumes of liquid."""

    def addFluidVolume(self, volume: FluidVolume) -> int:
        """
        Add a body of liquid under a free surface: a pool, a lake, the sea.

        Args:
            volume (FluidVolume): the liquid and where it is.

        Returns:
            int: its index.
        """

    def setFluidVolume(self, index: int, volume: FluidVolume) -> None:
        """
        Change a volume of liquid, for example its surface height or its current.

        Args:
            index (int): its index.
            volume (FluidVolume): the liquid and where it is.
        """

    def getFluidVolumes(self) -> list[FluidVolume]:
        """Get the volumes of liquid."""

    def clearFluids(self) -> None:
        """Remove every volume of liquid and the ambient fluid."""

    def setFluidCoefficients(self, object: Object, coefficients: FluidCoefficients) -> None:
        """
        Set how fluids act on every collision body of an object.

        Args:
            object (Object): the object.
            coefficients (FluidCoefficients): the coefficients.
        """

    def setERP(self, erp: float, erp2: float = 0) -> None:
        """Set the error reduction parameter (ERP)."""

    def set_contact_solver_parameters(self, alpha_init: float, alpha_min: float, alpha_decay: float, max_iters: int, threshold: float) -> None:
        """
        Set contact solver parameters.

        Args:
            alpha_init (float): alpha init.
            alpha_min (float): alpha minimum.
            alpha_decay (float): alpha decay.
            max_iters (float): maximum number of iterations.
            threshold (float): threshold.
        """

    def setContactSolverIterationOrder(self, order: bool) -> None: ...

    def setSensorNoiseSeed(self, arg: int, /) -> None: ...

    def getSensorNoiseSeed(self) -> int: ...

    def getWorldTime(self) -> float:
        """
        Return the total integrated time (which is updated at every `integrate2()`` call).

        Returns:
            float: world time.
        """

    def setWorldTime(self, time: float) -> None:
        """
        Set the world time.

        Args:
            time (float): world time
        """

    def getContactSolver(self) -> contact.BisectionContactSolver:
        """
        Return the bisection contact solver used.

        Returns:
            BisectionContactSolver: contact solver.
        """

    def rayTest(self, start: Annotated[NDArray[numpy.float64], dict(shape=(3), writable=False)], direction: Annotated[NDArray[numpy.float64], dict(shape=(3), writable=False)], length: float, closestOnly: bool = True, objectId: int = 18446744073709551606, localId: int = 18446744073709551606, collisionMask: int = 18446744073709551615) -> RayCollisionList:
        """
        Returns the internal reference of the ray collision list it
        contains the geoms (position, normal, object world/local id) and the number of intersections

        Args:
            start The start position of the ray.
            direction The direction of the ray.
            length The length of the ray.
            closestOnly Ignored. The current rayTest always stores the first collision.
            objectId ignores collisions with an object with the same objectId and localId
            localId ignores collisions with an object with the same objectId and localId
            collisionMask Collision mask to filter collisions. By default, it records collisions with all collision groups.
            @return A reference to the internal container which contains all ray collisions.
        """

    def addExplosion(self, position: Annotated[NDArray[numpy.float64], dict(order='C')], properties: ExplosionProperties = ...) -> int:
        """
        Set off an explosion at the current world time. Its blast wave pushes every dynamic body it reaches
        during the following steps: each patch of a body's collision geometry takes the pressure of the wave
        when the shock arrives there, surfaces facing the charge also take the reflected pressure and the blast
        wind, and bodies hidden behind others are spared both. A requested crater is dug at once.

        Args:
            position (np.array[float[3]]): position of the charge in the world frame (m).
            properties (ExplosionProperties): the charge and what it affects.

        Returns:
            int: identifier of the explosion.
        """

    def getExplosions(self) -> list[Explosion]:
        """
        Copies of the explosions set off most recently, oldest first: at most the last 256 and every one whose blast still spreads; identifiers increase.
        """

    def addProjectile(self, position: Annotated[NDArray[numpy.float64], dict(order='C')], velocity: Annotated[NDArray[numpy.float64], dict(order='C')], properties: ProjectileProperties = ...) -> int:
        """
        Launch a projectile at the current world time. Each step moves it under gravity and air drag and
        sweeps its path with a ray, so it cannot pass through a body. At the first body on its path it
        detonates or stops, handing its momentum to a dynamic body it hits; a time fuse detonates it in flight.

        Args:
            position (np.array[float[3]]): launch position in the world frame (m).
            velocity (np.array[float[3]]): launch velocity in the world frame (m/s).
            properties (ProjectileProperties): what the projectile is and does.

        Returns:
            int: identifier of the projectile.
        """

    def removeProjectile(self, id: int) -> bool:
        """Remove a projectile in flight; returns whether it was in flight."""

    def getProjectiles(self) -> list[Projectile]:
        """Copies of the projectiles in flight."""

    def getProjectileHits(self) -> list[ProjectileHit]:
        """
        Copies of the most recent projectile hits (at most 256), oldest first; identifiers increase.
        """

    def getProjectileLaunches(self) -> list[Projectile]:
        """
        Copies of the projectiles launched most recently, as launched (at most 256), oldest first, whatever became of them.
        """

    def getTendons(self) -> list[Tendon]: ...

    def getTendonCouplings(self) -> list[TendonCoupling]: ...

    def lockMutex(self) -> None: ...

    def unlockMutex(self) -> None: ...

    def lock(self) -> None: ...

    def unlock(self) -> None: ...

    def updateSelfCollisionCache(self) -> None: ...

    def addSegmentationClass(self, name: str) -> int: ...

    def getSegmentationClassLabel(self, name: str) -> int: ...

    def getSegmentationClassName(self, label: int) -> str: ...

    def getSegmentationClassNames(self) -> list[str]: ...

    def setSegmentationClassName(self, label: int, name: str) -> None: ...

    class Checkpoint:
        def __init__(self) -> None: ...

        def __bool__(self) -> bool: ...

        def getWorldTime(self) -> float: ...

    class SingleBodySnapshot:
        def __init__(self) -> None: ...

        @property
        def objectId(self) -> int: ...

        @property
        def objectType(self) -> ObjectType: ...

        @property
        def bodyType(self) -> BodyType: ...

        @bodyType.setter
        def bodyType(self, arg: BodyType, /) -> None: ...

        @property
        def material(self) -> str: ...

        @material.setter
        def material(self, arg: str, /) -> None: ...

        @property
        def collisionGroup(self) -> int: ...

        @collisionGroup.setter
        def collisionGroup(self, arg: int, /) -> None: ...

        @property
        def collisionMask(self) -> int: ...

        @collisionMask.setter
        def collisionMask(self, arg: int, /) -> None: ...

        @property
        def position(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @position.setter
        def position(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

        @property
        def quaternion(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @quaternion.setter
        def quaternion(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

        @property
        def linearVelocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @linearVelocity.setter
        def linearVelocity(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

        @property
        def angularVelocity(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

        @angularVelocity.setter
        def angularVelocity(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    class SleepSettings:
        @property
        def enabled(self) -> bool: ...

        @property
        def linearVelThreshold(self) -> float: ...

        @property
        def angularVelThreshold(self) -> float: ...

        @property
        def quietSteps(self) -> int: ...

    class CollisionCounters:
        @property
        def deformableParticleCandidates(self) -> int: ...

        @property
        def deformableParticleAabbTests(self) -> int: ...

        @property
        def deformableParticleContacts(self) -> int: ...

        @property
        def deformableParticleGridCells(self) -> int: ...

        @property
        def sweptCcdCandidates(self) -> int: ...

        @property
        def sweptCcdContacts(self) -> int: ...

    def captureCheckpoint(self) -> World.Checkpoint: ...

    def restoreCheckpoint(self, checkpoint: World.Checkpoint, raiseOnError: bool = False) -> bool:
        """
        Restore a checkpoint. Return false on rejection, or raise ValueError with raiseOnError=True.
        """

    def captureSingleBodySnapshot(self, object: SingleBodyObject) -> World.SingleBodySnapshot: ...

    @overload
    def restoreSingleBodySnapshot(self, object: SingleBodyObject, snapshot: World.SingleBodySnapshot, restoreCollisionProperties: bool = True) -> bool: ...

    @overload
    def restoreSingleBodySnapshot(self, id: int, snapshot: World.SingleBodySnapshot, restoreCollisionProperties: bool = True) -> bool: ...

    def cloneSingleBodyObject(self, source: SingleBodyObject, name: str = '') -> SingleBodyObject: ...

    def getObjectCount(self) -> int: ...

    def getObjectById(self, id: int) -> Object: ...

    def isObjectAlive(self, id: int) -> bool: ...

    def setObjectCollisionFilter(self, object: Object, collision_group: int, collision_mask: int, localIdx: int = 18446744073709551615) -> bool: ...

    def setObjectMaterial(self, object: Object, material: str, localIdx: int = 18446744073709551615) -> bool: ...

    def setContactSolverParam(self, alpha_init: float, alpha_min: float, alpha_decay: float, maxIter: int, threshold: float) -> None: ...

    def setMonolithicDeformables(self, enabled: bool) -> None: ...

    def isMonolithicDeformables(self) -> bool: ...

    def setSleepingEnabled(self, enabled: bool) -> None: ...

    def isSleepingEnabled(self) -> bool: ...

    def setSleepingParameters(self, linearVelThreshold: float, angularVelThreshold: float, quietSteps: int) -> None: ...

    def getSleepingParameters(self) -> World.SleepSettings: ...

    def setSleepingVelocityThresholds(self, linearVelThreshold: float, angularVelThreshold: float) -> None: ...

    def wakeObject(self, object: Object) -> None: ...

    def wakeAll(self) -> None: ...

    @overload
    def isObjectSleeping(self, object: Object) -> bool: ...

    @overload
    def isObjectSleeping(self, index: int) -> bool: ...

    def setContactSettings(self, settings: ContactSettings) -> None: ...

    def getContactSettings(self) -> ContactSettings: ...

    def setBroadphaseSettings(self, settings: BroadphaseSettings) -> None: ...

    def getBroadphaseSettings(self) -> BroadphaseSettings: ...

    def setCollisionCountersEnabled(self, enabled: bool) -> None: ...

    def getCollisionCountersEnabled(self) -> bool: ...

    def getLastCollisionCounters(self) -> World.CollisionCounters: ...

    def addUsdArticulatedSystem(self, usdPath: str, collision_group: int = 1, collision_mask: int = 18446744073709551615, options: ArticulatedSystemOption = ...) -> ArticulatedSystem: ...

    def exportMeshAssetsToObj(self, directory: str, prefix: str = 'mesh') -> list[str]: ...

    def loadMjcfFile(self, configFile: str) -> None: ...

    @overload
    def addDeformableCloth(self, vertices: Annotated[NDArray[numpy.float64], dict(order='C')], triangles: Sequence[Sequence[int]], pinnedVertices: Sequence[int] = [], material: DeformableObject.Material = ..., contactMaterial: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> DeformableObject: ...

    @overload
    def addDeformableCloth(self, fileName: str, scale: float = 1.0, pinnedVertices: Sequence[int] = [], material: DeformableObject.Material = ..., contactMaterial: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> DeformableObject: ...

    def addDeformableVolume(self, vertices: Annotated[NDArray[numpy.float64], dict(order='C')], tetrahedra: Sequence[Sequence[int]], pinnedVertices: Sequence[int] = [], material: DeformableObject.Material = ..., contactMaterial: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> DeformableObject: ...

    def addDeformableObject(self, fileName: str, material: DeformableObject.Material, options: DeformableObject.MeshBuildOptions, pinnedVertices: Sequence[int] = [], contactMaterial: str = 'default', collision_group: int = 1, collision_mask: int = 18446744073709551615) -> DeformableObject: ...

    def addGranularParticles(self, positions: Annotated[NDArray[numpy.float64], dict(order='C')], radii: Sequence[float], material: GranularSystem.Material = ...) -> GranularSystem: ...

    def addGranularBox(self, options: GranularSystem.BoxOptions, material: GranularSystem.Material = ...) -> GranularSystem: ...

    def addDeformableTerrain(self, xSamples: int, ySamples: int, xSize: float, ySize: float, centerX: float, centerY: float, heights: Sequence[float], soil: DeformableTerrain.Material = ..., material: str = 'default', collision_group: int = 9223372036854775808, collision_mask: int = 18446744073709551615) -> DeformableTerrain: ...

class BroadphaseType(enum.Enum):
    Sap3Axis = 1

    MultiBoxPrune = 2

class BroadphaseSettings:
    def __init__(self) -> None: ...

    @property
    def type(self) -> BroadphaseType: ...

    @type.setter
    def type(self, arg: BroadphaseType, /) -> None: ...

    @property
    def mbpUseWorldBounds(self) -> bool: ...

    @mbpUseWorldBounds.setter
    def mbpUseWorldBounds(self, arg: bool, /) -> None: ...

    @property
    def mbpPadding(self) -> float: ...

    @mbpPadding.setter
    def mbpPadding(self, arg: float, /) -> None: ...

    @property
    def mbpMaxCellsPerAxis(self) -> int: ...

    @mbpMaxCellsPerAxis.setter
    def mbpMaxCellsPerAxis(self, arg: int, /) -> None: ...

    @property
    def mbpMaxCellsPerObject(self) -> int: ...

    @mbpMaxCellsPerObject.setter
    def mbpMaxCellsPerObject(self, arg: int, /) -> None: ...

    @property
    def mbpWorldMin(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @mbpWorldMin.setter
    def mbpWorldMin(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def mbpWorldMax(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @mbpWorldMax.setter
    def mbpWorldMax(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def mbpCellSize(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @mbpCellSize.setter
    def mbpCellSize(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

class ContactSettings:
    def __init__(self) -> None: ...

    @property
    def gjkMaxIterations(self) -> int: ...

    @gjkMaxIterations.setter
    def gjkMaxIterations(self, arg: int, /) -> None: ...

    @property
    def gjkTolerance(self) -> float: ...

    @gjkTolerance.setter
    def gjkTolerance(self, arg: float, /) -> None: ...

    @property
    def epaMaxIterations(self) -> int: ...

    @epaMaxIterations.setter
    def epaMaxIterations(self, arg: int, /) -> None: ...

    @property
    def epaTolerance(self) -> float: ...

    @epaTolerance.setter
    def epaTolerance(self, arg: float, /) -> None: ...

    @property
    def maxContactsPerPair(self) -> int: ...

    @maxContactsPerPair.setter
    def maxContactsPerPair(self, arg: int, /) -> None: ...

    @property
    def sweptCcdEnabled(self) -> bool: ...

    @sweptCcdEnabled.setter
    def sweptCcdEnabled(self, arg: bool, /) -> None: ...

    @property
    def sweptCcdMinSpeed(self) -> float: ...

    @sweptCcdMinSpeed.setter
    def sweptCcdMinSpeed(self, arg: float, /) -> None: ...

    @property
    def sweptCcdSpeculativeMargin(self) -> float: ...

    @sweptCcdSpeculativeMargin.setter
    def sweptCcdSpeculativeMargin(self, arg: float, /) -> None: ...

    @property
    def broadphase(self) -> BroadphaseSettings: ...

    @broadphase.setter
    def broadphase(self, arg: BroadphaseSettings, /) -> None: ...

class RayCollisionItem:
    def getPosition(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]: ...

    def getNormal(self) -> Annotated[NDArray[numpy.float64], dict(shape=(3), order='C')]:
        """
        World-frame unit normal of the surface hit, pointing toward the side the ray came from.
        """

    def getObject(self) -> Object: ...

    def getCollisionBody(self) -> CollisionBodyHandle: ...

class RayCollisionList:
    def size(self) -> int: ...

    def __len__(self) -> int: ...

    def setSize(self, size: int) -> None: ...

    def resize(self, size: int) -> None: ...

    def reserve(self, size: int) -> None: ...

    def at(self, idx: int) -> RayCollisionItem: ...

    def __getitem__(self, arg: int, /) -> RayCollisionItem: ...

hasRayrai: bool = True

def __getattr__(arg: str, /) -> object: ...

from raisimpy_rayrai import RayraiWindow as RayraiWindow, RayraiViewer as RayraiViewer
