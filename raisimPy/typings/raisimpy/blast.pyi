

kAmbientPressure: float = 101325.0

kAmbientSoundSpeed: float = 340.3

kMinScaledDistance: float = 0.1

kMaxScaledDistance: float = 40.0

kSurfaceBurstFactor: float = 1.8

class Wave:
    """The blast at one distance from a charge."""

    def __init__(self) -> None: ...

    @property
    def overpressure(self) -> float:
        """Peak side-on overpressure (Pa)."""

    @overpressure.setter
    def overpressure(self, arg: float, /) -> None: ...

    @property
    def impulse(self) -> float:
        """Positive-phase side-on impulse per unit area (Pa s)."""

    @impulse.setter
    def impulse(self, arg: float, /) -> None: ...

    @property
    def arrivalTime(self) -> float:
        """Time from the detonation until the shock arrives (s)."""

    @arrivalTime.setter
    def arrivalTime(self, arg: float, /) -> None: ...

    @property
    def duration(self) -> float:
        """Duration of the positive phase (s)."""

    @duration.setter
    def duration(self, arg: float, /) -> None: ...

    @property
    def decay(self) -> float:
        """Decay coefficient b of the Friedlander pulse."""

    @decay.setter
    def decay(self, arg: float, /) -> None: ...

    @property
    def shockSpeed(self) -> float:
        """Speed of the shock front (m/s)."""

    @shockSpeed.setter
    def shockSpeed(self, arg: float, /) -> None: ...

    @property
    def reflectedOverpressure(self) -> float:
        """Peak overpressure on a surface facing the blast squarely (Pa)."""

    @reflectedOverpressure.setter
    def reflectedOverpressure(self, arg: float, /) -> None: ...

    @property
    def dynamicPressure(self) -> float:
        """Peak dynamic pressure of the blast wind (Pa)."""

    @dynamicPressure.setter
    def dynamicPressure(self, arg: float, /) -> None: ...

    def overpressureAt(self, sinceArrival: float) -> float:
        """
        Side-on overpressure (Pa) a time sinceArrival (s) after the shock arrived; zero outside the positive phase.
        """

def wave(distance: float, chargeMass: float) -> Wave:
    """The blast of chargeMass kg of TNT in free air at distance m."""

def reach(chargeMass: float) -> float:
    """Distance (m) at which a charge's blast stops loading objects."""
