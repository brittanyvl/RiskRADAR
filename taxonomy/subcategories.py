"""
taxonomy/subcategories.py
-------------------------
Level 2 sub-category definitions for hierarchical CICTT taxonomy.

This module defines sub-categories that extend the CICTT Level 1 categories
with more specific classification based on:
- Industry standards (IATA LOC-I, CFIT analysis frameworks)
- HFACS (Human Factors Analysis and Classification System)
- Technical sub-systems (FAA/EASA maintenance categorization)

INTELLECTUAL PROPERTY NOTE:
---------------------------
All definitions, keywords, and seed phrases in this file are ORIGINAL CONTENT
written for the RiskRADAR project. The "source" metadata indicates the
CONCEPTUAL FRAMEWORK that informed our categorization approach, not copied text.

- Category NAMES use standard aviation terminology (not copyrightable)
- DEFINITIONS are written in our own words
- KEYWORDS are technical aviation vocabulary (not copyrightable)
- SEED PHRASES are original example sentences for embedding classification

We cite the following as conceptual references (not content sources):
- IATA (2015, 2019). Loss of Control In-Flight Analysis Reports
- IATA (2018). Controlled Flight Into Terrain Analysis Report
- Shappell & Wiegmann (2000). HFACS DOT/FAA/AM-00/7 (US Government, public domain)
- EASA (2024). LOC-I Prevention Guidance (European public guidance)
- SKYbrary Aviation Safety Database (Source: www.skybrary.aero)
"""

from .cictt import CICTTCategory

# =============================================================================
# LOC-I (Loss of Control - Inflight) Sub-categories
# Conceptual framework informed by IATA LOC-I Analysis and EASA guidance
# All definitions, keywords, and seed phrases are original content
# =============================================================================

LOC_I_SUBCATEGORIES = [
    CICTTCategory(
        code="LOC-I-STALL",
        name="Aerodynamic Stall",
        description="Loss of control due to aerodynamic stall, including stall during all phases of flight, stall recovery failure, and stall warning system issues.",
        keywords=[
            "stall", "aerodynamic stall", "stall warning", "stick shaker",
            "stick pusher", "angle of attack", "alpha limit", "stall recovery",
            "stall spin", "approach to stall", "stall buffet", "airspeed decay",
            "critical angle of attack", "stall characteristics"
        ],
        seed_phrases=[
            "The airplane stalled during the approach due to inadequate airspeed",
            "Failure to recover from an aerodynamic stall during departure",
            "The stick shaker activated but the pilot failed to reduce angle of attack",
            "Inadvertent stall during maneuvering flight at low altitude"
        ],
        parent_code="LOC-I",
        level=2,
        metadata={"source": "IATA LOC-I Analysis"}
    ),
    CICTTCategory(
        code="LOC-I-UPSET",
        name="Aircraft Upset",
        description="Unusual aircraft attitude or flight path not intended by the pilot, including recovery from unusual attitudes, roll/pitch exceedances.",
        keywords=[
            "upset", "unusual attitude", "roll exceedance", "pitch exceedance",
            "nose high", "nose low", "bank angle", "inverted", "spiral dive",
            "attitude excursion", "aircraft upset recovery", "UPRT",
            "graveyard spiral", "steep bank"
        ],
        seed_phrases=[
            "The aircraft entered an unusual attitude with extreme pitch and roll",
            "Recovery from upset was not accomplished before impact with terrain",
            "The airplane entered a spiral dive from which the pilot did not recover",
            "Unintended nose-high unusual attitude followed by aerodynamic stall"
        ],
        parent_code="LOC-I",
        level=2,
        metadata={"source": "IATA LOC-I Analysis"}
    ),
    CICTTCategory(
        code="LOC-I-SD",
        name="Spatial Disorientation",
        description="Loss of control caused by pilot spatial disorientation, including somatogravic illusion, leans, and vestibular illusions.",
        keywords=[
            "spatial disorientation", "vestibular illusion", "leans",
            "somatogravic illusion", "vertigo", "disorientation", "illusion",
            "sensory illusion", "visual illusion", "false horizon",
            "black hole approach", "featureless terrain"
        ],
        seed_phrases=[
            "The pilot became spatially disoriented in instrument conditions",
            "Spatial disorientation led to loss of control during night flight",
            "The pilot experienced a somatogravic illusion during go-around",
            "Vestibular illusion resulted in uncontrolled descent into terrain"
        ],
        parent_code="LOC-I",
        level=2,
        metadata={"source": "HFACS/IATA"}
    ),
    CICTTCategory(
        code="LOC-I-ENV",
        name="Environmental LOC",
        description="Loss of control induced by environmental factors including turbulence, wind shear, wake turbulence, and icing effects on controllability.",
        keywords=[
            "turbulence", "wind shear", "wake turbulence", "mountain wave",
            "severe turbulence", "convective", "microburst", "downdraft",
            "rotor", "gusty winds", "crosswind", "tailwind shear",
            "atmospheric disturbance"
        ],
        seed_phrases=[
            "Loss of control after encountering severe turbulence",
            "The aircraft encountered a microburst during approach",
            "Wake turbulence from preceding aircraft caused loss of control",
            "Mountain wave rotor induced uncontrollable roll"
        ],
        parent_code="LOC-I",
        level=2,
        metadata={"source": "IATA LOC-I Analysis"}
    ),
    CICTTCategory(
        code="LOC-I-SYS",
        name="System-Induced LOC",
        description="Loss of control caused by flight control system failures, automation surprises, or autopilot malfunctions.",
        keywords=[
            "flight control", "autopilot", "autothrottle", "automation",
            "fly-by-wire", "control system failure", "servo", "actuator",
            "trim runaway", "flight control malfunction", "mode confusion",
            "automation surprise", "flight director"
        ],
        seed_phrases=[
            "The autopilot disconnected unexpectedly leading to loss of control",
            "Flight control system failure resulted in uncontrollable aircraft",
            "Trim runaway led to extreme pitch attitude and loss of control",
            "Mode confusion during automated approach led to loss of control"
        ],
        parent_code="LOC-I",
        level=2,
        metadata={"source": "EASA LOC-I Guidance"}
    ),
    CICTTCategory(
        code="LOC-I-LOAD",
        name="Loading/CG Issues",
        description="Loss of control due to improper loading, center of gravity out of limits, or cargo shift.",
        keywords=[
            "center of gravity", "CG", "weight and balance", "overweight",
            "cargo shift", "load shift", "aft CG", "forward CG",
            "improper loading", "out of balance", "payload", "fuel imbalance",
            "weight distribution"
        ],
        seed_phrases=[
            "The aircraft CG was outside aft limits causing loss of control",
            "Cargo shifted during flight resulting in uncontrollable pitch",
            "Overweight condition contributed to inability to recover from stall",
            "Improper fuel loading caused lateral control difficulties"
        ],
        parent_code="LOC-I",
        level=2,
        metadata={"source": "NTSB/FAA"}
    ),
]

# =============================================================================
# CFIT (Controlled Flight Into Terrain) Sub-categories
# Conceptual framework informed by IATA CFIT Analysis and SKYbrary (www.skybrary.aero)
# All definitions, keywords, and seed phrases are original content
# =============================================================================

CFIT_SUBCATEGORIES = [
    CICTTCategory(
        code="CFIT-NAV",
        name="Navigation Error",
        description="CFIT due to navigation errors including incorrect position, course deviation, or approach procedure errors.",
        keywords=[
            "navigation error", "position error", "course deviation",
            "wrong approach", "procedure error", "waypoint", "fix",
            "track error", "lateral deviation", "procedural error",
            "navigation display", "FMS error", "GPS", "VOR"
        ],
        seed_phrases=[
            "The crew flew the wrong approach procedure for the runway",
            "Navigation error led to aircraft striking terrain short of runway",
            "The pilot deviated from the published approach course",
            "Incorrect waypoint entry resulted in CFIT"
        ],
        parent_code="CFIT",
        level=2,
        metadata={"source": "IATA CFIT Analysis"}
    ),
    CICTTCategory(
        code="CFIT-SA",
        name="Situational Awareness",
        description="CFIT due to loss of situational awareness regarding terrain, altitude, or aircraft position.",
        keywords=[
            "situational awareness", "terrain awareness", "altitude awareness",
            "position awareness", "SA", "lost awareness", "distraction",
            "workload", "task saturation", "crew coordination",
            "monitoring", "attention"
        ],
        seed_phrases=[
            "The crew lost situational awareness of their proximity to terrain",
            "Pilot failed to maintain altitude awareness during approach",
            "Distraction led to loss of terrain awareness and impact",
            "High workload degraded situational awareness during descent"
        ],
        parent_code="CFIT",
        level=2,
        metadata={"source": "IATA CFIT Analysis"}
    ),
    CICTTCategory(
        code="CFIT-VIS",
        name="Visual Conditions/Illusions",
        description="CFIT in visual conditions due to visual illusions, black hole approach, or featureless terrain.",
        keywords=[
            "visual illusion", "black hole", "dark night", "featureless terrain",
            "sloping terrain", "rain on windscreen", "no horizon",
            "visual approach", "night visual", "unlighted terrain",
            "visual reference", "approach lighting"
        ],
        seed_phrases=[
            "The pilot experienced a black hole illusion during night approach",
            "Visual illusion from sloping terrain caused premature descent",
            "Featureless terrain at night led to controlled descent into water",
            "The crew made a visual approach without adequate lighting references"
        ],
        parent_code="CFIT",
        level=2,
        metadata={"source": "SKYbrary/IATA"}
    ),
    CICTTCategory(
        code="CFIT-TAWS",
        name="TAWS/GPWS Issues",
        description="CFIT involving terrain awareness warning system issues including non-equipment, inhibition, or ignored warnings.",
        keywords=[
            "TAWS", "GPWS", "ground proximity", "terrain warning",
            "warning ignored", "warning inhibited", "no TAWS", "not equipped",
            "pull up", "terrain terrain", "too low terrain",
            "escape maneuver"
        ],
        seed_phrases=[
            "The aircraft was not equipped with TAWS when it impacted terrain",
            "The crew ignored repeated GPWS pull up warnings",
            "TAWS warning came too late for effective terrain escape",
            "The crew inhibited the GPWS during approach"
        ],
        parent_code="CFIT",
        level=2,
        metadata={"source": "IATA CFIT Analysis"}
    ),
    CICTTCategory(
        code="CFIT-PROC",
        name="Procedural Deviation",
        description="CFIT due to deviation from standard operating procedures, stabilized approach criteria, or minimum altitude requirements.",
        keywords=[
            "procedural deviation", "SOP", "unstabilized", "below MDA",
            "below DA", "minimum altitude", "descent below", "continued approach",
            "go-around", "stabilized approach", "approach procedure",
            "briefing", "callout"
        ],
        seed_phrases=[
            "The crew continued descent below minimum altitude without visual contact",
            "Unstabilized approach continued to impact with terrain",
            "The pilot deviated from standard operating procedures during approach",
            "Failure to execute go-around at decision altitude led to CFIT"
        ],
        parent_code="CFIT",
        level=2,
        metadata={"source": "SKYbrary"}
    ),
]

# =============================================================================
# SCF-PP (System/Component Failure - Powerplant) Sub-categories
# Based on FAA maintenance categorization
# =============================================================================

SCF_PP_SUBCATEGORIES = [
    CICTTCategory(
        code="SCF-PP-ENG",
        name="Engine Mechanical Failure",
        description="Mechanical failure of engine components including turbine, compressor, bearings, and internal engine parts.",
        keywords=[
            "engine failure", "turbine blade", "compressor blade", "bearing",
            "engine seizure", "internal failure", "rotor burst", "disk",
            "shaft", "case breach", "uncontained", "fatigue crack",
            "engine destruction"
        ],
        seed_phrases=[
            "The engine failed due to a fractured high-pressure turbine blade",
            "Compressor blade liberation caused uncontained engine failure",
            "Engine bearing failure led to complete loss of power",
            "Fatigue crack in the engine disk resulted in engine destruction"
        ],
        parent_code="SCF-PP",
        level=2,
        metadata={"source": "FAA"}
    ),
    CICTTCategory(
        code="SCF-PP-FUEL",
        name="Fuel System Failure",
        description="Failures in fuel delivery system including fuel pumps, fuel control, contamination, and fuel leaks.",
        keywords=[
            "fuel system", "fuel pump", "fuel control", "fuel contamination",
            "water in fuel", "fuel leak", "fuel starvation", "fuel exhaustion",
            "fuel selector", "fuel quantity", "fuel management",
            "carburetor", "fuel injection"
        ],
        seed_phrases=[
            "Fuel contamination with water caused engine power loss",
            "Fuel pump failure resulted in loss of engine power",
            "Improper fuel management led to fuel exhaustion and forced landing",
            "Fuel system leak caused in-flight engine shutdown"
        ],
        parent_code="SCF-PP",
        level=2,
        metadata={"source": "FAA"}
    ),
    CICTTCategory(
        code="SCF-PP-PROP",
        name="Propeller/Rotor Failure",
        description="Failures of propeller, rotor, or rotorcraft drive system components.",
        keywords=[
            "propeller", "prop", "rotor", "blade", "hub", "pitch control",
            "feather", "governor", "rotor head", "mast", "swashplate",
            "tail rotor", "main rotor", "gearbox", "transmission"
        ],
        seed_phrases=[
            "The propeller blade separated due to fatigue cracking",
            "Tail rotor drive shaft failure led to loss of directional control",
            "Propeller governor malfunction caused uncommanded pitch change",
            "Main rotor gearbox failure resulted in autorotation"
        ],
        parent_code="SCF-PP",
        level=2,
        metadata={"source": "FAA"}
    ),
    CICTTCategory(
        code="SCF-PP-FIRE",
        name="Powerplant Fire",
        description="Fire or smoke originating from powerplant, nacelle, or engine compartment.",
        keywords=[
            "engine fire", "nacelle fire", "fire warning", "smoke",
            "fire loop", "fire detection", "fire suppression", "hot section",
            "oil leak", "fuel leak fire", "exhaust", "fire containment"
        ],
        seed_phrases=[
            "An engine fire developed due to a fuel line leak in the nacelle",
            "Fire warning light illuminated followed by engine shutdown",
            "Uncontained engine failure resulted in nacelle fire",
            "Oil leak ignited on hot engine components causing fire"
        ],
        parent_code="SCF-PP",
        level=2,
        metadata={"source": "FAA"}
    ),
]

# =============================================================================
# SCF-NP (System/Component Failure - Non-Powerplant) Sub-categories
# Based on FAA/EASA system categorization
# =============================================================================

SCF_NP_SUBCATEGORIES = [
    CICTTCategory(
        code="SCF-NP-FLT",
        name="Flight Control System",
        description="Failures of flight control systems including control surfaces, cables, pushrods, and actuators.",
        keywords=[
            "flight control", "aileron", "elevator", "rudder", "stabilizer",
            "control cable", "pushrod", "bellcrank", "control surface",
            "trim tab", "servo", "control jam", "control restriction",
            "flight control actuator"
        ],
        seed_phrases=[
            "The elevator control cable failed due to wear and corrosion",
            "Aileron flutter caused structural failure of control surface",
            "Horizontal stabilizer actuator malfunction led to loss of control",
            "Control surface separation occurred during flight"
        ],
        parent_code="SCF-NP",
        level=2,
        metadata={"source": "FAA/EASA"}
    ),
    CICTTCategory(
        code="SCF-NP-HYD",
        name="Hydraulic System",
        description="Failures of hydraulic systems including pumps, actuators, lines, and fluid loss.",
        keywords=[
            "hydraulic", "hydraulic failure", "hydraulic pump", "hydraulic line",
            "hydraulic fluid", "hydraulic leak", "hydraulic pressure",
            "brake", "landing gear hydraulic", "flight control hydraulic",
            "power control unit", "PCU"
        ],
        seed_phrases=[
            "Hydraulic system failure resulted in loss of flight control authority",
            "Hydraulic fluid leak led to loss of brake effectiveness",
            "Hydraulic pump failure caused degraded flight control response",
            "Landing gear failed to extend due to hydraulic system failure"
        ],
        parent_code="SCF-NP",
        level=2,
        metadata={"source": "FAA/EASA"}
    ),
    CICTTCategory(
        code="SCF-NP-ELEC",
        name="Electrical System",
        description="Failures of electrical systems including generators, batteries, wiring, and avionics.",
        keywords=[
            "electrical", "generator", "alternator", "battery", "wiring",
            "circuit breaker", "bus", "electrical fire", "electrical failure",
            "avionics", "instrument failure", "power loss electrical",
            "short circuit", "arc"
        ],
        seed_phrases=[
            "Electrical system failure resulted in loss of flight instruments",
            "Wiring short circuit caused electrical fire in the cockpit",
            "Generator failure led to loss of primary electrical power",
            "Battery failure during night flight caused loss of lighting"
        ],
        parent_code="SCF-NP",
        level=2,
        metadata={"source": "FAA/EASA"}
    ),
    CICTTCategory(
        code="SCF-NP-STRUCT",
        name="Structural Failure",
        description="Failures of aircraft structure including wings, fuselage, empennage, and attachments.",
        keywords=[
            "structural", "wing", "fuselage", "empennage", "spar",
            "skin", "rivet", "fatigue", "corrosion", "crack", "separation",
            "delamination", "in-flight breakup", "structural failure",
            "attachment"
        ],
        seed_phrases=[
            "Fatigue cracking in the wing spar led to in-flight structural failure",
            "Corrosion weakened the fuselage skin resulting in decompression",
            "Empennage separation occurred due to improper maintenance",
            "In-flight structural failure of the horizontal stabilizer"
        ],
        parent_code="SCF-NP",
        level=2,
        metadata={"source": "FAA/EASA"}
    ),
    CICTTCategory(
        code="SCF-NP-GEAR",
        name="Landing Gear System",
        description="Failures of landing gear systems including gear collapse, retraction issues, and brake failures.",
        keywords=[
            "landing gear", "gear collapse", "nose gear", "main gear",
            "gear retraction", "gear extension", "brake", "tire",
            "wheel", "strut", "gear door", "gear indication",
            "gear lock"
        ],
        seed_phrases=[
            "The nose landing gear collapsed during the landing roll",
            "Landing gear failed to extend for landing",
            "Main gear tire failure during takeoff led to runway departure",
            "Brake failure caused runway overrun during landing"
        ],
        parent_code="SCF-NP",
        level=2,
        metadata={"source": "FAA/EASA"}
    ),
]

# =============================================================================
# Human Factors Sub-categories (Applied to relevant L1 categories)
# Based on HFACS - Human Factors Analysis and Classification System
# Reference: Shappell & Wiegmann (2000). DOT/FAA/AM-00/7
# =============================================================================

HFACS_SUBCATEGORIES = [
    CICTTCategory(
        code="HF-SKILL",
        name="Skill-Based Error",
        description="Errors in highly practiced, automatic behaviors. Includes slips (attention failures) and lapses (memory failures).",
        keywords=[
            "skill error", "slip", "lapse", "attention failure", "memory failure",
            "inadvertent", "omission", "forgot", "missed", "overlooked",
            "procedure step skipped", "automatic behavior", "routine error",
            "checklist missed"
        ],
        seed_phrases=[
            "The pilot inadvertently moved the flap lever instead of the gear lever",
            "A checklist item was missed during the preflight",
            "The pilot forgot to set the altimeter to the local pressure",
            "Attention lapse resulted in missed radio call"
        ],
        parent_code=None,  # Applied to multiple parents
        level=2,
        hfacs_type="SKILL",
        metadata={"source": "HFACS"}
    ),
    CICTTCategory(
        code="HF-DECISION",
        name="Decision Error",
        description="Intentional behaviors that proceed as planned but the plan is inadequate. Includes poor decision-making, misdiagnosis, and rule-based errors.",
        keywords=[
            "decision error", "poor judgment", "decision", "elected to",
            "chose to", "decided to", "judgment", "planning", "risk",
            "misdiagnosis", "wrong decision", "rule-based error",
            "inadequate plan"
        ],
        seed_phrases=[
            "The pilot elected to continue the approach in deteriorating weather",
            "Poor decision-making led to flight into known icing conditions",
            "The captain decided to continue despite fuel concerns",
            "Misdiagnosis of the engine problem led to incorrect corrective action"
        ],
        parent_code=None,
        level=2,
        hfacs_type="DECISION",
        metadata={"source": "HFACS"}
    ),
    CICTTCategory(
        code="HF-PERCEPTUAL",
        name="Perceptual Error",
        description="Errors due to degraded sensory input or misperception. Includes visual illusions, spatial disorientation, and misread instruments.",
        keywords=[
            "perceptual error", "misread", "misinterpret", "illusion",
            "visual", "perception", "sensory", "misperceived", "saw",
            "thought", "believed", "interpreted", "misjudged",
            "depth perception"
        ],
        seed_phrases=[
            "The pilot misread the altimeter leading to premature descent",
            "Visual illusion caused the pilot to perceive incorrect aircraft attitude",
            "The pilot misjudged the distance to the runway threshold",
            "Instrument misinterpretation led to incorrect aircraft configuration"
        ],
        parent_code=None,
        level=2,
        hfacs_type="PERCEPTUAL",
        metadata={"source": "HFACS"}
    ),
    CICTTCategory(
        code="HF-VIOLATION",
        name="Violation",
        description="Willful disregard of rules and regulations. Includes routine violations (habitual), exceptional violations (isolated), and reckless behavior.",
        keywords=[
            "violation", "willful", "disregard", "intentional", "regulatory",
            "reckless", "negligent", "unauthorized", "prohibited",
            "contrary to", "in violation of", "non-compliance",
            "deviated from"
        ],
        seed_phrases=[
            "The pilot willfully flew below the minimum safe altitude",
            "The crew violated sterile cockpit procedures",
            "The pilot operated the aircraft in violation of VFR weather minimums",
            "Reckless maneuvering at low altitude led to the accident"
        ],
        parent_code=None,
        level=2,
        hfacs_type="VIOLATION",
        metadata={"source": "HFACS"}
    ),
]

# =============================================================================
# ICE (Icing) Sub-categories
# =============================================================================

ICE_SUBCATEGORIES = [
    CICTTCategory(
        code="ICE-STRUCT",
        name="Structural Icing",
        description="Icing on aircraft external surfaces including wings, tail, propeller, and other aerodynamic surfaces.",
        keywords=[
            "structural ice", "wing ice", "tail ice", "propeller ice",
            "airframe icing", "ice accretion", "rime ice", "clear ice",
            "mixed ice", "deice", "anti-ice", "ice protection",
            "leading edge ice"
        ],
        seed_phrases=[
            "Structural icing on the wings caused loss of lift and control",
            "Tail icing led to pitch control problems during approach",
            "Severe rime ice accretion exceeded deicing system capability",
            "Propeller ice caused vibration and power loss"
        ],
        parent_code="ICE",
        level=2,
        metadata={"source": "FAA/NTSB"}
    ),
    CICTTCategory(
        code="ICE-INDUCT",
        name="Induction Icing",
        description="Icing in engine air induction systems including carburetor ice, intake ice, and filter ice.",
        keywords=[
            "carburetor ice", "carb ice", "induction ice", "intake ice",
            "throttle ice", "carburetor heat", "alternate air",
            "intake icing", "filter ice", "engine induction"
        ],
        seed_phrases=[
            "Carburetor icing caused engine power loss during cruise",
            "The pilot failed to apply carburetor heat in icing conditions",
            "Induction system icing led to engine failure",
            "Air filter icing restricted engine airflow"
        ],
        parent_code="ICE",
        level=2,
        metadata={"source": "FAA"}
    ),
    CICTTCategory(
        code="ICE-PITOT",
        name="Pitot/Static Icing",
        description="Icing of pitot-static system causing erroneous airspeed, altitude, or vertical speed indications.",
        keywords=[
            "pitot ice", "static ice", "pitot heat", "pitot tube",
            "airspeed indication", "altitude indication", "unreliable airspeed",
            "erroneous indication", "iced over", "blocked pitot"
        ],
        seed_phrases=[
            "Pitot icing caused erroneous airspeed indications",
            "Failure to activate pitot heat led to blocked pitot tube",
            "Unreliable airspeed due to pitot-static system icing",
            "Iced static port caused erroneous altitude readings"
        ],
        parent_code="ICE",
        level=2,
        metadata={"source": "FAA/NTSB"}
    ),
]

# =============================================================================
# FUEL Sub-categories
# =============================================================================

FUEL_SUBCATEGORIES = [
    CICTTCategory(
        code="FUEL-EXHAUST",
        name="Fuel Exhaustion",
        description="Complete depletion of usable fuel due to inadequate fuel planning or fuel leak.",
        keywords=[
            "fuel exhaustion", "ran out of fuel", "fuel depleted", "no fuel",
            "tanks empty", "fuel planning", "fuel required", "reserve",
            "endurance", "fuel remaining"
        ],
        seed_phrases=[
            "The engines quit due to fuel exhaustion",
            "Inadequate fuel planning resulted in fuel exhaustion",
            "The aircraft ran out of fuel short of the destination",
            "Fuel exhaustion occurred due to underestimated fuel consumption"
        ],
        parent_code="FUEL",
        level=2,
        metadata={"source": "NTSB"}
    ),
    CICTTCategory(
        code="FUEL-STARVE",
        name="Fuel Starvation",
        description="Engine fuel starvation with fuel remaining in tanks due to fuel system mismanagement or malfunction.",
        keywords=[
            "fuel starvation", "fuel selector", "fuel management",
            "crossfeed", "fuel tank", "tank selection", "fuel imbalance",
            "unusable fuel", "fuel system", "fuel switch"
        ],
        seed_phrases=[
            "Fuel starvation occurred due to improper fuel tank selection",
            "The pilot failed to switch fuel tanks causing engine stoppage",
            "Fuel imbalance led to fuel starvation despite fuel remaining",
            "Fuel selector valve malfunction caused fuel starvation"
        ],
        parent_code="FUEL",
        level=2,
        metadata={"source": "NTSB"}
    ),
    CICTTCategory(
        code="FUEL-CONTAM",
        name="Fuel Contamination",
        description="Fuel contamination including water, debris, wrong fuel type, or biological growth.",
        keywords=[
            "fuel contamination", "water in fuel", "contaminated fuel",
            "wrong fuel", "misfuel", "debris", "sediment", "biological",
            "fuel quality", "fuel sample", "sumping"
        ],
        seed_phrases=[
            "Water contamination in the fuel caused engine failure",
            "The aircraft was misfueled with jet fuel instead of avgas",
            "Contaminated fuel led to loss of engine power",
            "Failure to sump fuel tanks resulted in water-induced engine failure"
        ],
        parent_code="FUEL",
        level=2,
        metadata={"source": "FAA/NTSB"}
    ),
]

# =============================================================================
# WSTRW (Windshear or Thunderstorm) Sub-categories
# =============================================================================

WSTRW_SUBCATEGORIES = [
    CICTTCategory(
        code="WSTRW-MICRO",
        name="Microburst",
        description="Encounter with microburst including wet and dry microbursts during takeoff, approach, or landing.",
        keywords=[
            "microburst", "downdraft", "outflow", "wind shear",
            "performance decreasing", "airspeed loss", "sink rate",
            "reactive windshear", "predictive windshear"
        ],
        seed_phrases=[
            "The aircraft encountered a microburst during approach",
            "Microburst downdraft caused impact short of the runway",
            "Performance-decreasing wind shear from microburst",
            "Airspeed and altitude loss due to microburst encounter"
        ],
        parent_code="WSTRW",
        level=2,
        metadata={"source": "FAA"}
    ),
    CICTTCategory(
        code="WSTRW-TSTORM",
        name="Thunderstorm Encounter",
        description="Flight into or near thunderstorm cells including associated turbulence, hail, and lightning.",
        keywords=[
            "thunderstorm", "convective", "cumulonimbus", "CB", "cell",
            "hail", "lightning", "storm", "convection", "severe weather",
            "weather radar", "storm cell", "weather avoidance"
        ],
        seed_phrases=[
            "The aircraft penetrated a severe thunderstorm cell",
            "Thunderstorm turbulence caused structural damage",
            "Hail from a thunderstorm caused engine damage",
            "Attempted penetration of a thunderstorm line"
        ],
        parent_code="WSTRW",
        level=2,
        metadata={"source": "FAA/NTSB"}
    ),
]

# =============================================================================
# Aggregate Collections
# =============================================================================

# All subcategories in a single list
ALL_SUBCATEGORIES = (
    LOC_I_SUBCATEGORIES +
    CFIT_SUBCATEGORIES +
    SCF_PP_SUBCATEGORIES +
    SCF_NP_SUBCATEGORIES +
    HFACS_SUBCATEGORIES +
    ICE_SUBCATEGORIES +
    FUEL_SUBCATEGORIES +
    WSTRW_SUBCATEGORIES
)

# Mapping from parent code to subcategories
PARENT_TO_SUBCATEGORIES: dict[str, list[CICTTCategory]] = {
    "LOC-I": LOC_I_SUBCATEGORIES,
    "CFIT": CFIT_SUBCATEGORIES,
    "SCF-PP": SCF_PP_SUBCATEGORIES,
    "SCF-NP": SCF_NP_SUBCATEGORIES,
    "ICE": ICE_SUBCATEGORIES,
    "FUEL": FUEL_SUBCATEGORIES,
    "WSTRW": WSTRW_SUBCATEGORIES,
}

# HFACS subcategories can apply to multiple parent categories
HFACS_APPLICABLE_PARENTS = ["LOC-I", "CFIT", "FUEL", "UIMC", "LALT"]

# Lookup by code
SUBCATEGORY_BY_CODE: dict[str, CICTTCategory] = {
    cat.code: cat for cat in ALL_SUBCATEGORIES
}


def get_subcategories_for_parent(parent_code: str) -> list[CICTTCategory]:
    """
    Get all subcategories for a given parent CICTT code.

    Args:
        parent_code: The Level 1 CICTT category code (e.g., "LOC-I")

    Returns:
        List of subcategory CICTTCategory objects
    """
    return PARENT_TO_SUBCATEGORIES.get(parent_code, [])


def get_hfacs_subcategories() -> list[CICTTCategory]:
    """
    Get all HFACS-type subcategories.

    These can be applied across multiple parent categories
    where human factors are relevant.
    """
    return HFACS_SUBCATEGORIES


def has_subcategories(parent_code: str) -> bool:
    """Check if a parent category has defined subcategories."""
    return parent_code in PARENT_TO_SUBCATEGORIES


def get_subcategory_count() -> dict[str, int]:
    """Get count of subcategories per parent."""
    return {
        parent: len(subs)
        for parent, subs in PARENT_TO_SUBCATEGORIES.items()
    }
