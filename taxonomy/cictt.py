"""
taxonomy/cictt.py
-----------------
CAST/ICAO Common Taxonomy Team (CICTT) occurrence categories.

Based on CICTT Aviation Occurrence Categories v4.7 (December 2017).
Reference: https://www.ntsb.gov/safety/data/Documents/datafiles/OccurrenceCategoryDefinitions.pdf

These categories are used internationally for classifying aviation accidents
and incidents by their primary causal/occurrence type.

Extended to support hierarchical taxonomy:
- Level 1: CICTT occurrence categories (27 categories)
- Level 2: Sub-categories (industry-specific or HFACS-based)

INTELLECTUAL PROPERTY NOTE:
---------------------------
CICTT is an international PUBLIC STANDARD developed by CAST/ICAO specifically
for industry-wide adoption. The NTSB reference document is a US Government
publication and is PUBLIC DOMAIN (17 U.S.C. § 105).

Category codes and names follow the CICTT standard. All descriptions, keywords,
and seed phrases in this file are ORIGINAL CONTENT written for RiskRADAR.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class CICTTCategory:
    """
    A CICTT occurrence category with matching criteria.

    Supports both Level 1 (CICTT) and Level 2 (sub-categories) in hierarchy.

    Attributes:
        code: Unique category identifier (e.g., "LOC-I" or "LOC-I-STALL")
        name: Human-readable category name
        description: Detailed description of what this category covers
        keywords: Keywords for text matching
        seed_phrases: Example phrases for embedding similarity
        parent_code: Code of parent category (None for L1, parent code for L2)
        level: Hierarchy level (1 for CICTT, 2 for sub-categories)
        is_active: Whether this category is active for classification
        hfacs_type: HFACS type if this is an HFACS sub-category (SKILL, DECISION, etc.)
        metadata: Additional metadata for extensibility
    """
    code: str
    name: str
    description: str
    keywords: list[str]  # Keywords for text matching
    seed_phrases: list[str]  # Example phrases for embedding similarity
    parent_code: Optional[str] = None  # For subcategories
    level: int = 1  # 1 = CICTT Level 1, 2 = Sub-category
    is_active: bool = True  # Enable/disable categories
    hfacs_type: Optional[str] = None  # SKILL, DECISION, PERCEPTUAL, VIOLATION
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extensibility


# CICTT Occurrence Categories - ordered by typical frequency/importance
CICTT_CATEGORIES = [
    CICTTCategory(
        code="LOC-I",
        name="Loss of Control - Inflight",
        description="Loss of aircraft control while airborne. Can result from aerodynamic stall, spatial disorientation, aircraft system failures, or environmental factors.",
        keywords=[
            "loss of control", "stall", "spin", "spiral", "uncontrolled descent",
            "spatial disorientation", "unusual attitude", "aerodynamic stall",
            "angle of attack", "airspeed decay", "stick shaker", "stick pusher",
            "unrecoverable", "departed controlled flight", "loss of lift"
        ],
        seed_phrases=[
            "The airplane entered an aerodynamic stall and the pilot was unable to recover",
            "Loss of control during flight due to pilot spatial disorientation",
            "The aircraft departed controlled flight following a stall",
            "Uncontrolled descent into terrain after loss of lift"
        ]
    ),
    CICTTCategory(
        code="CFIT",
        name="Controlled Flight Into Terrain",
        description="Collision with terrain, water, or obstacle while aircraft is under control. Typically involves pilot awareness issues, navigation errors, or inadequate terrain clearance.",
        keywords=[
            "controlled flight into terrain", "cfit", "impact terrain",
            "flew into", "struck terrain", "mountain", "obstacle", "tower",
            "terrain clearance", "minimum safe altitude", "gpws", "taws",
            "terrain awareness", "ground proximity"
        ],
        seed_phrases=[
            "The aircraft flew into rising terrain during approach",
            "Controlled flight into terrain during visual approach at night",
            "The crew failed to maintain adequate terrain clearance",
            "Impact with mountainous terrain during instrument conditions"
        ]
    ),
    CICTTCategory(
        code="SCF-PP",
        name="System/Component Failure - Powerplant",
        description="Failure or malfunction of powerplant (engine), propeller, rotor, or drive train components.",
        keywords=[
            "engine failure", "power loss", "engine malfunction", "propeller",
            "rotor", "gearbox", "transmission", "turbine", "compressor",
            "fuel system", "engine fire", "power plant", "thrust reverser",
            "engine shutdown", "flameout", "mechanical failure engine"
        ],
        seed_phrases=[
            "The engine failed due to a fractured turbine blade",
            "Loss of engine power during climb due to fuel contamination",
            "Propeller separation due to fatigue crack in blade",
            "Total loss of engine power following mechanical failure"
        ]
    ),
    CICTTCategory(
        code="SCF-NP",
        name="System/Component Failure - Non-Powerplant",
        description="Failure or malfunction of aircraft systems not related to powerplant, including flight controls, hydraulics, electrical, and structural components.",
        keywords=[
            "system failure", "component failure", "malfunction", "structural",
            "flight control", "hydraulic", "electrical", "landing gear",
            "flap", "rudder", "elevator", "aileron", "autopilot", "instrument",
            "fatigue", "corrosion", "design defect", "manufacturing defect"
        ],
        seed_phrases=[
            "The flight control system malfunctioned due to a design defect",
            "Structural failure of the horizontal stabilizer",
            "Loss of hydraulic pressure resulted in control difficulties",
            "Landing gear collapse during touchdown"
        ]
    ),
    CICTTCategory(
        code="RE",
        name="Runway Excursion",
        description="Aircraft veers off or overruns the runway surface during takeoff or landing operations.",
        keywords=[
            "runway excursion", "overrun", "veer off", "runway departure",
            "departed runway", "off runway", "runway overshoot", "stopping distance",
            "brake failure", "hydroplaning", "contaminated runway", "wet runway"
        ],
        seed_phrases=[
            "The aircraft overran the runway end during landing",
            "The airplane veered off the runway during takeoff roll",
            "Runway excursion due to hydroplaning on wet runway",
            "Unable to stop on remaining runway distance"
        ]
    ),
    CICTTCategory(
        code="ARC",
        name="Abnormal Runway Contact",
        description="Landing or takeoff involving abnormal contact with runway surface, including hard landings, tail strikes, and gear-up landings.",
        keywords=[
            "hard landing", "tail strike", "gear up landing", "nose gear collapse",
            "bounced landing", "porpoise", "wheelbarrow", "wing strike",
            "prop strike", "abnormal touchdown", "heavy landing"
        ],
        seed_phrases=[
            "The aircraft made a hard landing exceeding design limits",
            "Tail strike during rotation due to improper technique",
            "Gear-up landing after pilot failed to extend landing gear",
            "The landing gear collapsed during touchdown"
        ]
    ),
    CICTTCategory(
        code="WSTRW",
        name="Wind Shear/Thunderstorm",
        description="Flight into wind shear, microburst, or thunderstorm conditions affecting aircraft control or performance.",
        keywords=[
            "wind shear", "microburst", "thunderstorm", "convective",
            "downdraft", "updraft", "gust", "turbulence severe",
            "weather penetration", "storm", "cumulonimbus"
        ],
        seed_phrases=[
            "The aircraft encountered a microburst on final approach",
            "Wind shear during takeoff resulted in performance degradation",
            "Flight into severe thunderstorm conditions",
            "Microburst-induced loss of airspeed during approach"
        ]
    ),
    CICTTCategory(
        code="TURB",
        name="Turbulence Encounter",
        description="Encounter with atmospheric turbulence including clear air turbulence, mountain wave, or wake turbulence.",
        keywords=[
            "turbulence", "clear air turbulence", "cat", "mountain wave",
            "wake turbulence", "wake vortex", "rough air", "chop",
            "severe turbulence", "extreme turbulence"
        ],
        seed_phrases=[
            "The aircraft encountered severe clear air turbulence",
            "Wake turbulence encounter behind heavy aircraft",
            "Mountain wave turbulence during cruise",
            "Injuries resulted from unexpected turbulence"
        ]
    ),
    CICTTCategory(
        code="ICE",
        name="Icing",
        description="Accumulation of ice on aircraft surfaces affecting performance or control, or engine icing affecting power.",
        keywords=[
            "icing", "ice accumulation", "airframe ice", "carburetor ice",
            "induction ice", "pitot ice", "windshield ice", "frost",
            "freezing rain", "supercooled", "anti-ice", "de-ice", "ice protection"
        ],
        seed_phrases=[
            "Ice accumulation on the wings resulted in loss of lift",
            "Carburetor icing caused engine power loss",
            "The aircraft encountered icing conditions beyond capability",
            "Failure to activate anti-ice systems"
        ]
    ),
    CICTTCategory(
        code="FUEL",
        name="Fuel Related",
        description="Fuel exhaustion, starvation, contamination, or management issues affecting aircraft operation.",
        keywords=[
            "fuel exhaustion", "fuel starvation", "out of fuel", "fuel management",
            "fuel contamination", "wrong fuel", "fuel leak", "fuel tank",
            "fuel selector", "fuel transfer", "fuel calculation", "fuel planning"
        ],
        seed_phrases=[
            "Fuel exhaustion due to inadequate fuel planning",
            "Engine failure due to fuel starvation from mismanaged tanks",
            "Contaminated fuel caused engine malfunction",
            "The pilot failed to switch fuel tanks"
        ]
    ),
    CICTTCategory(
        code="GCOL",
        name="Ground Collision",
        description="Collision between aircraft and another aircraft, vehicle, person, or obstacle while on the ground.",
        keywords=[
            "ground collision", "taxiway collision", "ramp collision",
            "vehicle collision", "wing tip strike", "ground vehicle",
            "collided on ground", "struck while taxiing", "hit vehicle"
        ],
        seed_phrases=[
            "The aircraft collided with a ground vehicle during taxi",
            "Wing tip struck another aircraft on the ramp",
            "Ground collision with service vehicle",
            "Taxiing aircraft struck parked aircraft"
        ]
    ),
    CICTTCategory(
        code="LOC-G",
        name="Loss of Control - Ground",
        description="Loss of aircraft control while on the ground during taxi, takeoff roll, or landing roll.",
        keywords=[
            "loss of control ground", "ground loop", "directional control",
            "runway departure", "veer off", "wheelbarrow", "nose over",
            "tailwheel", "crosswind landing", "brake asymmetry"
        ],
        seed_phrases=[
            "The pilot lost directional control during landing roll",
            "Ground loop during taxi in crosswind conditions",
            "Loss of control on contaminated taxiway",
            "The aircraft veered off during takeoff roll"
        ]
    ),
    CICTTCategory(
        code="UIMC",
        name="Unintended Flight in IMC",
        description="Unplanned encounter with instrument meteorological conditions by VFR pilot or aircraft.",
        keywords=[
            "vfr into imc", "inadvertent imc", "weather encounter",
            "visibility", "ceiling", "fog", "clouds", "instrument conditions",
            "vfr pilot", "visual flight rules", "weather deterioration"
        ],
        seed_phrases=[
            "The VFR pilot inadvertently entered instrument conditions",
            "Continued VFR flight into deteriorating weather",
            "Unintended flight into IMC resulted in spatial disorientation",
            "The non-instrument rated pilot encountered IMC"
        ]
    ),
    CICTTCategory(
        code="BIRD",
        name="Bird Strike",
        description="Collision or near collision with birds affecting aircraft operation.",
        keywords=[
            "bird strike", "bird ingestion", "bird hit", "geese", "flock",
            "bird remains", "windshield bird", "engine bird ingestion"
        ],
        seed_phrases=[
            "The aircraft struck a flock of birds during takeoff",
            "Bird ingestion into engine caused power loss",
            "Multiple bird strikes during approach",
            "Windshield penetrated by large bird"
        ]
    ),
    CICTTCategory(
        code="WILD",
        name="Wildlife Strike",
        description="Collision with wildlife other than birds.",
        keywords=[
            "wildlife strike", "deer", "animal", "wildlife on runway",
            "struck animal", "animal collision"
        ],
        seed_phrases=[
            "The aircraft struck a deer during takeoff roll",
            "Wildlife on runway resulted in collision",
            "Animal strike during landing"
        ]
    ),
    CICTTCategory(
        code="F-NI",
        name="Fire/Smoke - Non-Impact",
        description="Fire or smoke not resulting from impact, including in-flight fire or ground fire.",
        keywords=[
            "fire", "smoke", "in-flight fire", "cabin fire", "electrical fire",
            "engine fire", "cargo fire", "smoke in cockpit", "fumes"
        ],
        seed_phrases=[
            "In-flight fire in the cargo compartment",
            "Electrical fire in the cockpit",
            "Smoke filled the cabin during flight",
            "Engine fire during cruise flight"
        ]
    ),
    CICTTCategory(
        code="F-POST",
        name="Fire/Smoke - Post-Impact",
        description="Fire or smoke occurring as result of impact.",
        keywords=[
            "post-impact fire", "post crash fire", "fire after impact",
            "burned after", "fire following"
        ],
        seed_phrases=[
            "Post-impact fire consumed the aircraft",
            "Fire erupted following the crash",
            "The aircraft burned after impact with terrain"
        ]
    ),
    CICTTCategory(
        code="RI-VAP",
        name="Runway Incursion - Vehicle/Aircraft/Person",
        description="Incorrect presence of vehicle, aircraft, or person on runway.",
        keywords=[
            "runway incursion", "runway intrusion", "runway conflict",
            "occupied runway", "crossed runway", "hold short violation"
        ],
        seed_phrases=[
            "Vehicle crossed runway without clearance",
            "Aircraft landed on occupied runway",
            "Runway incursion by taxiing aircraft",
            "Near collision due to runway incursion"
        ]
    ),
    CICTTCategory(
        code="MAC",
        name="Midair Collision",
        description="Collision between aircraft while airborne.",
        keywords=[
            "midair collision", "mid-air", "collided in flight",
            "collision airborne", "struck in flight", "near midair"
        ],
        seed_phrases=[
            "The two aircraft collided in flight",
            "Midair collision during traffic pattern",
            "Near midair collision between aircraft"
        ]
    ),
    CICTTCategory(
        code="CTOL",
        name="Collision with Obstacle During Takeoff/Landing",
        description="Collision with obstacles during takeoff or landing phases.",
        keywords=[
            "struck obstacle", "hit trees", "struck wires", "power lines",
            "obstacle takeoff", "obstacle landing", "terrain clearance",
            "building strike", "antenna", "tower"
        ],
        seed_phrases=[
            "The aircraft struck power lines during approach",
            "Collision with trees during takeoff climb",
            "Failed to clear obstacles on departure",
            "Struck antenna during landing approach"
        ]
    ),
    CICTTCategory(
        code="ADRM",
        name="Aerodrome",
        description="Occurrences related to aerodrome design, service, or functionality.",
        keywords=[
            "aerodrome", "airport", "runway condition", "lighting",
            "marking", "approach aids", "obstacle", "pavement"
        ],
        seed_phrases=[
            "Inadequate runway lighting contributed to the accident",
            "Runway condition was not as reported",
            "Airport design contributed to the incident"
        ]
    ),
    CICTTCategory(
        code="ATM",
        name="ATM/CNS",
        description="Occurrences involving Air Traffic Management or Communication/Navigation/Surveillance service issues.",
        keywords=[
            "atc", "air traffic control", "controller", "communication",
            "navigation", "surveillance", "radar", "clearance", "separation"
        ],
        seed_phrases=[
            "Air traffic control error contributed to the accident",
            "Loss of radar contact with the aircraft",
            "Communication failure between pilot and ATC",
            "Inadequate separation provided by controller"
        ]
    ),
    CICTTCategory(
        code="CABIN",
        name="Cabin Safety Events",
        description="Miscellaneous occurrences in the passenger cabin.",
        keywords=[
            "cabin", "passenger", "flight attendant", "seatbelt",
            "overhead bin", "galley", "lavatory", "evacuation"
        ],
        seed_phrases=[
            "Passenger injured during turbulence",
            "Cabin crew injury during service",
            "Evacuation-related injuries"
        ]
    ),
    CICTTCategory(
        code="RAMP",
        name="Ground Handling",
        description="Occurrences during ground handling operations.",
        keywords=[
            "ground handling", "ramp", "pushback", "towing", "loading",
            "baggage", "cargo loading", "refueling", "de-icing ground"
        ],
        seed_phrases=[
            "Damage during pushback operations",
            "Ground handling error during loading",
            "Refueling incident on ramp"
        ]
    ),
    CICTTCategory(
        code="SEC",
        name="Security Related",
        description="Security-related occurrences including hijacking, interference, or sabotage.",
        keywords=[
            "security", "hijack", "interference", "unruly passenger",
            "sabotage", "threat", "bomb", "explosive"
        ],
        seed_phrases=[
            "Hijacking of the aircraft",
            "Unruly passenger interfered with crew",
            "Security threat aboard aircraft"
        ]
    ),
    CICTTCategory(
        code="AMAN",
        name="Abrupt Maneuver",
        description="Intentional abrupt maneuvering of aircraft by flight crew.",
        keywords=[
            "abrupt maneuver", "evasive action", "avoidance maneuver",
            "pull-up", "tcas ra", "resolution advisory"
        ],
        seed_phrases=[
            "Abrupt maneuver to avoid collision",
            "TCAS resolution advisory maneuver",
            "Evasive action caused passenger injuries"
        ]
    ),
    CICTTCategory(
        code="NAV",
        name="Navigation Errors",
        description="Incorrect navigation of aircraft.",
        keywords=[
            "navigation error", "wrong runway", "wrong airport",
            "lost", "off course", "navigation display"
        ],
        seed_phrases=[
            "The pilot landed at the wrong airport",
            "Navigation error resulted in airspace deviation",
            "Wrong runway selected for landing"
        ]
    ),
    CICTTCategory(
        code="LALT",
        name="Low Altitude Operations",
        description="Occurrences during intentional low altitude operations.",
        keywords=[
            "low altitude", "crop dusting", "aerial application",
            "survey", "pipeline patrol", "wire strike low"
        ],
        seed_phrases=[
            "Wire strike during agricultural operation",
            "Crash during low-altitude aerial survey",
            "Collision during crop dusting operation"
        ]
    ),
    CICTTCategory(
        code="OTHR",
        name="Other",
        description="Occurrences not covered by other categories.",
        keywords=["other"],
        seed_phrases=["Occurrence not fitting other categories"]
    ),
    CICTTCategory(
        code="UNK",
        name="Unknown",
        description="Insufficient information to categorize occurrence.",
        keywords=["unknown", "undetermined", "insufficient information"],
        seed_phrases=["Cause could not be determined"]
    ),
]

# Create lookup dictionaries
CICTT_BY_CODE = {cat.code: cat for cat in CICTT_CATEGORIES}
CICTT_CODES = [cat.code for cat in CICTT_CATEGORIES]


def get_category(code: str) -> CICTTCategory | None:
    """Get a CICTT category by its code."""
    return CICTT_BY_CODE.get(code.upper())


def get_all_categories() -> list[CICTTCategory]:
    """Get all CICTT categories."""
    return CICTT_CATEGORIES.copy()


def get_primary_categories() -> list[CICTTCategory]:
    """Get primary categories (excluding OTHR and UNK)."""
    return [cat for cat in CICTT_CATEGORIES if cat.code not in ("OTHR", "UNK")]
