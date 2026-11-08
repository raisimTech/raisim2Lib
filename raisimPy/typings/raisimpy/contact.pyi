from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

import raisimpy


class Contact:
    """Raisim Contact."""

    def getPosition(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the contact position.

        Returns:
            np.array[float[3]]: contact position in the world.
        """

    def get_position(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    def getNormal(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the contact normal.

        Returns:
            np.array[float[3]]: contact normal in the world.
        """

    def getContactFrame(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the contact frame.

        Returns:
            np.array[float[3, 3]]: contact frame.
        """

    def getIndexContactProblem(self) -> int:
        """
        Get the index contact problem.

        Returns:
            int: index.
        """

    def getPairObjectIndex(self) -> int:
        """
        Get the pair object index.

        Returns:
            int: index.
        """

    def getPairContactIndexInPairObject(self) -> int:
        """
        Get the pair contact index in pair objects.

        Returns:
            int: index.
        """

    def getIndexInObjectContactList(self) -> int:
        """
        Get the index of this contact in the object's contact list.

        Returns:
            int: index.
        """

    def getImpulse(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the impulse.

        Returns:
            np.array[float[3]]: impulse.
        """

    def getInvInertia(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Get the inverse apparent inertia.

        Returns:
            np.array[float[3,3]]: inverse apparent inertia.
        """

    def isObjectA(self) -> bool:
        """
        Check if it is object A.

        Returns:
            bool: True if object A is in contact.
        """

    def getPairObjectBodyType(self) -> raisimpy.BodyType:
        """
        Get the pair object body type.

        Returns:
            raisim.BodyType: the body type (STATIC, KINEMATIC, DYNAMIC)
        """

    def getlocalBodyIndex(self) -> int:
        """
        Get local body index.

        Returns:
            int: local body index.
        """

    def getDepth(self) -> float:
        """
        Get the depth.

        Returns:
            float: depth.
        """

    def isSelfCollision(self) -> bool:
        """
        Return True if self-collision is enabled.

        Returns:
            bool: True if self-collision is enabled.
        """

    def skip(self) -> bool:
        """
        Return True if we contact is skipped.

        Returns:
            bool: True if the contact is skipped.
        """

    def getCollisionBodyA(self) -> raisimpy.CollisionBodyHandle: ...

    def getCollisionBodyB(self) -> raisimpy.CollisionBodyHandle: ...

class AABB:
    """Contact engine axis-aligned bounding box."""

    def __init__(self) -> None: ...

    @property
    def min(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @min.setter
    def min(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def max(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @max.setter
    def max(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    def overlaps(self, other: AABB) -> bool: ...

class Single3DContactProblem:
    def __init__(self) -> None: ...

    @property
    def imp_i(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @imp_i.setter
    def imp_i(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def velInit(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @velInit.setter
    def velInit(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def MappInv_i(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @MappInv_i.setter
    def MappInv_i(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def Mapp_i(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @Mapp_i.setter
    def Mapp_i(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def Mapp_iInv22(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @Mapp_iInv22.setter
    def Mapp_iInv22(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def Mapp_i22(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @Mapp_i22.setter
    def Mapp_i22(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def basisMat(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @basisMat.setter
    def basisMat(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def axis(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @axis.setter
    def axis(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def mu(self) -> float: ...

    @mu.setter
    def mu(self, arg: float, /) -> None: ...

    @property
    def n2_mu(self) -> float: ...

    @n2_mu.setter
    def n2_mu(self, arg: float, /) -> None: ...

    @property
    def muinv(self) -> float: ...

    @muinv.setter
    def muinv(self, arg: float, /) -> None: ...

    @property
    def coeffRes(self) -> float: ...

    @coeffRes.setter
    def coeffRes(self, arg: float, /) -> None: ...

    @property
    def bounceThres(self) -> float: ...

    @bounceThres.setter
    def bounceThres(self, arg: float, /) -> None: ...

    @property
    def bouceVel(self) -> float: ...

    @bouceVel.setter
    def bouceVel(self, arg: float, /) -> None: ...

    @property
    def Mapp_iInv11(self) -> float: ...

    @Mapp_iInv11.setter
    def Mapp_iInv11(self, arg: float, /) -> None: ...

    @property
    def impact_vel(self) -> float: ...

    @impact_vel.setter
    def impact_vel(self, arg: float, /) -> None: ...

    @property
    def depth(self) -> float: ...

    @depth.setter
    def depth(self, arg: float, /) -> None: ...

    @property
    def tempE(self) -> float: ...

    @tempE.setter
    def tempE(self, arg: float, /) -> None: ...

    @property
    def rank(self) -> int: ...

    @rank.setter
    def rank(self, arg: int, /) -> None: ...

    @property
    def pointIdA(self) -> int: ...

    @pointIdA.setter
    def pointIdA(self, arg: int, /) -> None: ...

    @property
    def jointId(self) -> int: ...

    @jointId.setter
    def jointId(self, arg: int, /) -> None: ...

    @property
    def pointIdB(self) -> int: ...

    @pointIdB.setter
    def pointIdB(self, arg: int, /) -> None: ...

    @property
    def position_W(self) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @position_W.setter
    def position_W(self, arg: Annotated[NDArray[numpy.float64], dict(order='C')], /) -> None: ...

    @property
    def mu_static(self) -> float: ...

    @mu_static.setter
    def mu_static(self, arg: float, /) -> None: ...

    @property
    def mu_static_vel_thresh(self) -> float: ...

    @mu_static_vel_thresh.setter
    def mu_static_vel_thresh(self, arg: float, /) -> None: ...

    @property
    def mu_static_vel_thresh_inv(self) -> float: ...

    @mu_static_vel_thresh_inv.setter
    def mu_static_vel_thresh_inv(self, arg: float, /) -> None: ...

    @property
    def obA(self) -> raisimpy.Object: ...

    @property
    def obB(self) -> raisimpy.Object: ...

class SolverConfiguration:
    def __init__(self) -> None: ...

    @property
    def alpha_init(self) -> float: ...

    @alpha_init.setter
    def alpha_init(self, arg: float, /) -> None: ...

    @property
    def alpha_low(self) -> float: ...

    @alpha_low.setter
    def alpha_low(self, arg: float, /) -> None: ...

    @property
    def alpha_decay(self) -> float: ...

    @alpha_decay.setter
    def alpha_decay(self, arg: float, /) -> None: ...

    @property
    def error_to_terminate(self) -> float: ...

    @error_to_terminate.setter
    def error_to_terminate(self, arg: float, /) -> None: ...

    @property
    def erp(self) -> float: ...

    @erp.setter
    def erp(self, arg: float, /) -> None: ...

    @property
    def erp2(self) -> float: ...

    @erp2.setter
    def erp2(self, arg: float, /) -> None: ...

    @property
    def maxIteration(self) -> int: ...

    @maxIteration.setter
    def maxIteration(self, arg: int, /) -> None: ...

class BisectionContactSolver:
    def __init__(self) -> None: ...

    def solve(self, contact_problems: Sequence[Single3DContactProblem]) -> list[Single3DContactProblem]: ...

    def updateConfig(self, config: SolverConfiguration) -> None: ...

    def setTimestep(self, dt: float) -> None: ...

    def setOrder(self, order: bool) -> None: ...

    def getLoopCounter(self) -> int: ...

    def getConfig(self) -> SolverConfiguration: ...
