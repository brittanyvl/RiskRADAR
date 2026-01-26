# RiskRADAR Level 2 Taxonomy: Sources, Standards, and Terminology

A comprehensive technical report documenting the industry standards and academic sources underlying the RiskRADAR hierarchical accident classification system.

---

## Attribution Notice

This document references content from **SKYbrary Aviation Safety** (www.skybrary.aero), a free electronic repository of aviation safety knowledge maintained by EUROCONTROL, ICAO, and the Flight Safety Foundation. SKYbrary content is used in accordance with their [License Agreement](https://skybrary.aero/licence-agreement-and-code-conduct).

**Source: www.skybrary.aero**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Taxonomy Architecture Overview](#taxonomy-architecture-overview)
3. [Level 1: CICTT Foundation](#level-1-cictt-foundation)
4. [Level 2: Subcategory Sources](#level-2-subcategory-sources)
   - [LOC-I Subcategories (IATA/EASA)](#loc-i-subcategories-iata-easa)
   - [CFIT Subcategories (IATA/SKYbrary)](#cfit-subcategories-iata-skybrary)
   - [SCF-PP/SCF-NP Subcategories (FAA/EASA)](#scf-ppscf-np-subcategories-faa-easa)
   - [HFACS Human Factors (DOT/FAA)](#hfacs-human-factors-dot-faa)
   - [Environmental Subcategories (ICE, FUEL, WSTRW)](#environmental-subcategories)
5. [Key Terminology Glossary](#key-terminology-glossary)
6. [Intellectual Property & Original Content Statement](#intellectual-property--original-content-statement)
7. [References](#references)

---

## Executive Summary

The RiskRADAR taxonomy uses a **two-level hierarchical classification** to categorize aviation accidents:

| Level | Source | Categories | Purpose |
|-------|--------|------------|---------|
| **Level 1** | CICTT v4.7 | 27 categories | Occurrence type (what happened) |
| **Level 2** | Multiple standards | 32 subcategories | Specific mechanism (how/why it happened) |

This design combines:
- **International consensus** (CICTT is used globally by ICAO, FAA, EASA, NTSB)
- **Domain-specific research** (IATA analysis reports for LOC-I and CFIT)
- **Human factors science** (HFACS framework from FAA research)
- **Technical system taxonomy** (FAA/EASA maintenance categorization)

---

## Taxonomy Architecture Overview

```
Level 1: CICTT Occurrence Category (27 total)
    │
    ├── LOC-I (Loss of Control - Inflight)
    │       ├── LOC-I-STALL  (Aerodynamic Stall)
    │       ├── LOC-I-UPSET  (Aircraft Upset)
    │       ├── LOC-I-SD     (Spatial Disorientation)
    │       ├── LOC-I-ENV    (Environmental LOC)
    │       ├── LOC-I-SYS    (System-Induced LOC)
    │       └── LOC-I-LOAD   (Loading/CG Issues)
    │
    ├── CFIT (Controlled Flight Into Terrain)
    │       ├── CFIT-NAV     (Navigation Error)
    │       ├── CFIT-SA      (Situational Awareness)
    │       ├── CFIT-VIS     (Visual Conditions/Illusions)
    │       ├── CFIT-TAWS    (TAWS/GPWS Issues)
    │       └── CFIT-PROC    (Procedural Deviation)
    │
    ├── SCF-PP (System/Component Failure - Powerplant)
    │       ├── SCF-PP-ENG   (Engine Mechanical)
    │       ├── SCF-PP-FUEL  (Fuel System)
    │       ├── SCF-PP-PROP  (Propeller/Rotor)
    │       └── SCF-PP-FIRE  (Powerplant Fire)
    │
    ├── SCF-NP (System/Component Failure - Non-Powerplant)
    │       ├── SCF-NP-FLT   (Flight Control System)
    │       ├── SCF-NP-HYD   (Hydraulic System)
    │       ├── SCF-NP-ELEC  (Electrical System)
    │       ├── SCF-NP-STRUCT (Structural Failure)
    │       └── SCF-NP-GEAR  (Landing Gear System)
    │
    ├── ICE (Icing)
    │       ├── ICE-STRUCT   (Structural Icing)
    │       ├── ICE-INDUCT   (Induction Icing)
    │       └── ICE-PITOT    (Pitot/Static Icing)
    │
    ├── FUEL (Fuel Related)
    │       ├── FUEL-EXHAUST (Fuel Exhaustion)
    │       ├── FUEL-STARVE  (Fuel Starvation)
    │       └── FUEL-CONTAM  (Fuel Contamination)
    │
    ├── WSTRW (Windshear/Thunderstorm)
    │       ├── WSTRW-MICRO  (Microburst)
    │       └── WSTRW-TSTORM (Thunderstorm Encounter)
    │
    └── Human Factors (cross-cutting, from HFACS)
            ├── HF-SKILL     (Skill-Based Error)
            ├── HF-DECISION  (Decision Error)
            ├── HF-PERCEPTUAL (Perceptual Error)
            └── HF-VIOLATION (Violation)
```

---

## Level 1: CICTT Foundation

### What is CICTT?

**CICTT** (CAST/ICAO Common Taxonomy Team) is a collaborative international effort to standardize aviation safety terminology. It was established in the early 2000s to address a critical problem: different countries and organizations were using inconsistent terminology to classify accidents, making global safety analysis impossible.

### Organization

CICTT is maintained jointly by:
- **CAST** (Commercial Aviation Safety Team) - US industry/government partnership
- **ICAO** (International Civil Aviation Organization) - UN aviation authority

### Version History

| Version | Year | Changes |
|---------|------|---------|
| 1.0 | 2006 | Initial 22 categories |
| 4.6 | 2013 | Expanded to 28 categories |
| **4.7** | 2017 | Current version (30 categories, 28 active) |

### Primary Reference

> **NTSB. (2013).** *Aviation Occurrence Categories: Definitions and Usage Notes (Version 4.6)*. National Transportation Safety Board.
> URL: https://www.ntsb.gov/safety/data/Documents/datafiles/OccurrenceCategoryDefinitions.pdf

This document provides:
- Official definitions for each category
- Usage notes explaining when to apply each category
- Examples of accidents that fit each category

### How CICTT is Used in RiskRADAR

We use the CICTT framework as Level 1 because:
1. **Industry standard**: The same categories are used by NTSB, FAA, ICAO, and airlines worldwide
2. **Proven on real data**: The categories were developed from analysis of thousands of actual accidents
3. **Semantic coherence**: Each category has clear definitions with minimal overlap
4. **Expert validation**: Developed by aviation safety professionals, not statistical algorithms

---

## Level 2: Subcategory Sources

### LOC-I Subcategories (IATA/EASA)

**Loss of Control - Inflight (LOC-I)** is the leading cause of fatal accidents in commercial aviation. Extensive research has identified specific mechanisms that lead to LOC-I events.

#### Primary Sources

1. **IATA. (2015).** *Loss of Control In-Flight Accident Analysis Report (1st Edition)*. International Air Transport Association.
   - URL: https://flightsafety.org/wp-content/uploads/2017/07/IATA-LOC-I-1st-Ed-2015.pdf
   - Analyzed 35 LOC-I fatal accidents (2001-2014)
   - Identified three major contributing factor categories

2. **IATA. (2019).** *Loss of Control In-Flight Accident Analysis Report*. International Air Transport Association.
   - URL: https://www.iata.org/contentassets/b6eb2adc248c484192101edd1ed36015/loc-i_2019.pdf
   - Updated analysis with additional accidents
   - Introduced Threat and Error Management (TEM) framework

3. **EASA. (2024).** *Loss of Control (LOC-I)*. European Union Aviation Safety Agency.
   - URL: https://www.easa.europa.eu/en/domains/general-aviation/flying-safely/loss-of-control
   - Prevention-focused guidance
   - Identifies specific LOC-I sub-types

#### LOC-I Subcategory Definitions

| Code | Name | Source | Description |
|------|------|--------|-------------|
| **LOC-I-STALL** | Aerodynamic Stall | IATA | Loss of lift due to exceeding critical angle of attack. Includes approach-to-stall events, stall recovery failures, and stall warning issues. |
| **LOC-I-UPSET** | Aircraft Upset | IATA | Unintended aircraft attitude beyond normal operating envelope. Includes nose-high/nose-low situations, excessive bank angles, and spiral dives. |
| **LOC-I-SD** | Spatial Disorientation | HFACS/IATA | Loss of awareness of aircraft position and motion. Includes vestibular illusions (leans, somatogravic), visual illusions, and disorientation in IMC. |
| **LOC-I-ENV** | Environmental LOC | IATA | Loss of control induced by atmospheric conditions. Includes turbulence encounters, wind shear, wake turbulence, and mountain wave. |
| **LOC-I-SYS** | System-Induced LOC | EASA | Loss of control caused by aircraft system failures or automation issues. Includes autopilot disconnects, trim runaways, and mode confusion. |
| **LOC-I-LOAD** | Loading/CG Issues | NTSB/FAA | Loss of control due to improper loading. Includes out-of-envelope CG, overweight conditions, and cargo shift. |

#### Key Finding from IATA Research

The IATA analysis found that **pilot-induced factors** account for the majority of LOC-I accidents:
- 52% involved inappropriate pilot inputs
- 26% involved environmental factors
- 22% involved system/aircraft factors

This informed our decision to include HFACS human factors as cross-cutting subcategories.

---

### CFIT Subcategories (IATA/SKYbrary)

**Controlled Flight Into Terrain (CFIT)** occurs when an airworthy aircraft under crew control is flown into terrain, water, or obstacles with inadequate awareness.

#### Primary Sources

1. **IATA. (2018).** *Controlled Flight Into Terrain Accident Analysis Report (2008-2017 Data)*. International Air Transport Association.
   - URL: https://www.iata.org/contentassets/06377898f60c46028a4dd38f13f979ad/cfit-report.pdf
   - Analyzed commercial aviation CFIT accidents over 10 years
   - Identified five primary contributing factor categories

2. **SKYbrary. (2024).** *Controlled Flight Into Terrain (CFIT)*. SKYbrary Aviation Safety.
   - URL: https://skybrary.aero/articles/controlled-flight-terrain-cfit
   - Comprehensive database entry
   - Links to case studies and prevention strategies

3. **FAA. (2022).** *Controlled Flight Into Terrain*. Federal Aviation Administration.
   - URL: https://www.faa.gov/sites/faa.gov/files/2022-01/Controlled%20Flight%20into%20Terrain.pdf
   - General aviation focused
   - Emphasizes TAWS effectiveness

#### CFIT Subcategory Definitions

| Code | Name | Source | Description |
|------|------|--------|-------------|
| **CFIT-NAV** | Navigation Error | IATA | CFIT caused by navigation mistakes including wrong approach, course deviation, waypoint errors, or procedure misinterpretation. |
| **CFIT-SA** | Situational Awareness | IATA | CFIT due to loss of awareness of terrain proximity, altitude, or aircraft position. Often involves distraction or high workload. |
| **CFIT-VIS** | Visual Conditions/Illusions | SKYbrary/IATA | CFIT in visual conditions caused by visual illusions such as black hole approaches, sloping terrain illusions, or featureless terrain at night. |
| **CFIT-TAWS** | TAWS/GPWS Issues | IATA | CFIT involving Terrain Awareness Warning System issues including no equipment, inhibited warnings, or ignored alerts. |
| **CFIT-PROC** | Procedural Deviation | SKYbrary | CFIT due to deviation from standard procedures including descent below minimums, unstabilized approaches, or failure to execute go-around. |

#### Key Finding from IATA Research

TAWS (Terrain Awareness and Warning System) is highly effective:
- 100% of fatal CFIT accidents in the study involved either no TAWS or ignored/inhibited TAWS warnings
- When TAWS warnings were heeded, terrain escape was successful

---

### SCF-PP/SCF-NP Subcategories (FAA/EASA)

System and Component Failures are categorized into **Powerplant (SCF-PP)** and **Non-Powerplant (SCF-NP)** to distinguish between propulsion system failures and other aircraft systems.

#### Primary Sources

1. **FAA Maintenance Categories** - The FAA's aircraft certification and maintenance regulations define system categorization for airworthiness directives and service bulletins.

2. **EASA CS-25/CS-23** - European certification specifications define system categories for large and small aircraft.

#### SCF-PP Subcategory Definitions

| Code | Name | Description |
|------|------|-------------|
| **SCF-PP-ENG** | Engine Mechanical Failure | Failures of internal engine components: turbine blades, compressor, bearings, shafts. Includes uncontained failures. |
| **SCF-PP-FUEL** | Fuel System Failure | Failures in fuel delivery: pumps, control units, contamination, leaks affecting engine operation. |
| **SCF-PP-PROP** | Propeller/Rotor Failure | Failures of propeller, rotor, or drive system: blade separation, governor malfunction, gearbox failure. |
| **SCF-PP-FIRE** | Powerplant Fire | Fire or smoke originating from engine, nacelle, or engine compartment. |

#### SCF-NP Subcategory Definitions

| Code | Name | Description |
|------|------|-------------|
| **SCF-NP-FLT** | Flight Control System | Failures of primary flight controls: control surfaces, cables, pushrods, actuators, trim systems. |
| **SCF-NP-HYD** | Hydraulic System | Failures of hydraulic systems: pumps, lines, actuators affecting flight controls, brakes, or landing gear. |
| **SCF-NP-ELEC** | Electrical System | Failures of electrical systems: generators, batteries, wiring, avionics, causing loss of critical functions. |
| **SCF-NP-STRUCT** | Structural Failure | Failures of aircraft structure: fatigue cracking, corrosion, in-flight breakup, delamination. |
| **SCF-NP-GEAR** | Landing Gear System | Failures of landing gear: collapse, extension/retraction issues, brake failures, tire failures. |

---

### HFACS Human Factors (DOT/FAA)

**HFACS (Human Factors Analysis and Classification System)** is a framework developed specifically for investigating aviation accidents. It provides a structured approach to identifying human error beyond simple "pilot error" attributions.

#### Primary Sources

1. **Shappell, S.A. & Wiegmann, D.A. (2000).** *The Human Factors Analysis and Classification System (HFACS)*. DOT/FAA/AM-00/7. Federal Aviation Administration.
   - URL: https://rosap.ntl.bts.gov/view/dot/21482
   - Original FAA research paper
   - Defines the 4-level HFACS framework

2. **Wiegmann, D.A. & Shappell, S.A. (2003).** *A Human Error Approach to Aviation Accident Analysis*. Ashgate Publishing.
   - Academic textbook expanding on HFACS
   - Detailed category definitions and examples

3. **SKYbrary. (2024).** *Human Factors Analysis and Classification System (HFACS)*. SKYbrary Aviation Safety.
   - URL: https://skybrary.aero/articles/human-factors-analysis-and-classification-system-hfacs
   - Summary and application guidance

#### HFACS Framework Structure

HFACS has four levels (we use only Level 1 - Unsafe Acts for subcategorization):

```
Level 4: Organizational Influences (resource management, climate, process)
    │
Level 3: Unsafe Supervision (inadequate supervision, planned inappropriate operations)
    │
Level 2: Preconditions for Unsafe Acts (environmental factors, crew state, crew factors)
    │
Level 1: Unsafe Acts (errors and violations) ← Used in RiskRADAR L2
```

#### HFACS Level 1 Subcategory Definitions

| Code | Name | HFACS Type | Description |
|------|------|------------|-------------|
| **HF-SKILL** | Skill-Based Error | Slip/Lapse | Errors in highly practiced, automatic behaviors. **Slips** are attention failures (wrong switch), **Lapses** are memory failures (forgot checklist item). |
| **HF-DECISION** | Decision Error | Mistake | Intentional behaviors where the plan is inadequate. Includes poor judgment, misdiagnosis, and proceeding despite known risks. |
| **HF-PERCEPTUAL** | Perceptual Error | Perception | Errors due to degraded sensory input or misinterpretation. Includes misread instruments, visual illusions, misjudged distances. |
| **HF-VIOLATION** | Violation | Willful | Willful disregard of rules. **Routine violations** are habitual shortcuts. **Exceptional violations** are isolated rule-breaking. |

#### Why HFACS is Applied Cross-Categorically

Human factors can contribute to many accident types. For example:
- **LOC-I** + **HF-DECISION**: Pilot elected to continue flight into icing conditions
- **CFIT** + **HF-PERCEPTUAL**: Pilot misread altimeter during approach
- **FUEL** + **HF-SKILL**: Pilot forgot to switch fuel tanks

We apply HFACS subcategories to these L1 parents: `LOC-I`, `CFIT`, `FUEL`, `UIMC`, `LALT`

---

### Environmental Subcategories

#### ICE (Icing) Subcategories

| Code | Name | Description |
|------|------|-------------|
| **ICE-STRUCT** | Structural Icing | Ice accumulation on external surfaces (wings, tail, propeller) affecting lift and control. Includes rime, clear, and mixed ice. |
| **ICE-INDUCT** | Induction Icing | Ice in engine air intake systems. Carburetor ice is most common, affecting piston engines in humid conditions. |
| **ICE-PITOT** | Pitot/Static Icing | Ice blocking pitot tubes or static ports, causing erroneous airspeed or altitude indications. |

**Source**: FAA Advisory Circulars on aircraft icing, NTSB accident analyses

#### FUEL Subcategories

| Code | Name | Description |
|------|------|-------------|
| **FUEL-EXHAUST** | Fuel Exhaustion | Complete depletion of all usable fuel. Caused by inadequate planning, unexpected headwinds, or fuel leaks. |
| **FUEL-STARVE** | Fuel Starvation | Engine fuel starvation with fuel remaining in tanks. Caused by improper tank selection, blocked lines, or system failures. |
| **FUEL-CONTAM** | Fuel Contamination | Fuel contaminated with water, debris, wrong fuel type, or biological growth causing engine malfunction. |

**Source**: NTSB accident data, FAA Safety Briefs

#### WSTRW (Windshear/Thunderstorm) Subcategories

| Code | Name | Description |
|------|------|-------------|
| **WSTRW-MICRO** | Microburst | Encounter with microburst—a localized column of sinking air producing strong downdrafts and outflow winds. Causes rapid airspeed/altitude loss. |
| **WSTRW-TSTORM** | Thunderstorm Encounter | Flight into thunderstorm cells causing severe turbulence, hail, lightning, or structural damage. |

**Source**: FAA windshear training materials, NTSB weather-related accident analyses

---

## Key Terminology Glossary

### Aviation Safety Terms

| Term | Definition |
|------|------------|
| **Angle of Attack (AOA)** | The angle between the wing chord line and the relative wind. Exceeding the critical AOA causes aerodynamic stall. |
| **CG (Center of Gravity)** | The point where the aircraft's weight is concentrated. Must remain within limits for safe flight. |
| **GPWS** | Ground Proximity Warning System. Earlier-generation terrain warning system that alerts based on radio altitude and descent rate. |
| **IMC** | Instrument Meteorological Conditions. Weather conditions requiring flight solely by reference to instruments. |
| **MDA/DA** | Minimum Descent Altitude / Decision Altitude. The lowest altitude on an approach where descent can continue without visual reference. |
| **SOP** | Standard Operating Procedures. Published procedures that flight crews must follow. |
| **Stick Shaker** | A stall warning device that vibrates the control column when approaching stall. |
| **TAWS** | Terrain Awareness and Warning System. Modern terrain warning system using GPS and terrain database. |
| **TEM** | Threat and Error Management. Framework for managing threats (environmental) and errors (crew) before they lead to undesired aircraft states. |
| **UPRT** | Upset Prevention and Recovery Training. Specialized training for recognizing and recovering from unusual attitudes. |
| **VMC** | Visual Meteorological Conditions. Weather conditions allowing flight by visual reference. |
| **VFR** | Visual Flight Rules. Flight rules requiring visual reference to terrain and other aircraft. |

### HFACS Terms

| Term | Definition |
|------|------------|
| **Slip** | An attention failure where the correct action is intended but incorrectly executed (e.g., moving wrong lever). |
| **Lapse** | A memory failure where an intended action is forgotten or not completed (e.g., skipping checklist item). |
| **Mistake** | A planning failure where the intended action is incorrect for the situation (e.g., misdiagnosing a problem). |
| **Violation** | A willful deviation from rules or procedures, either routine (habitual) or exceptional (isolated). |

---

## Intellectual Property & Original Content Statement

### How Sources Are Used

This taxonomy system was **developed independently** using the following approach:

| Source Type | How We Use It | What We Created |
|-------------|---------------|-----------------|
| **CICTT (Public Standard)** | Category codes and names directly | Our classification system implementation |
| **IATA Research** | Conceptual framework for subcategorization | Original definitions, keywords, seed phrases |
| **HFACS (Public Domain)** | Level 1 Unsafe Acts framework | Adapted definitions for aviation context |
| **SKYbrary** | Reference material and validation | Original descriptions with proper attribution |
| **FAA/NTSB/EASA** | Public domain guidance | Technical vocabulary and domain knowledge |

### Original Content

The following elements in our taxonomy are **entirely original**:

1. **Subcategory Definitions**: All description text is written in our own words
2. **Keywords**: Compiled from technical aviation vocabulary (not copyrightable)
3. **Seed Phrases**: Example sentences created for embedding-based classification
4. **Classification Logic**: Our embedding similarity approach is novel

### IATA Usage Clarification

We cite IATA research reports as the **conceptual foundation** for LOC-I and CFIT subcategorization. This means:

- ✅ We acknowledge IATA identified these subcategory patterns in their research
- ✅ We cite their reports as authoritative sources
- ✅ We wrote our own definitions, not copied from IATA documents
- ❌ We do NOT reproduce IATA's copyrighted analysis text, tables, or figures
- ❌ We do NOT redistribute IATA publications

This is consistent with standard academic and professional practice: using published research to inform one's own work while providing proper attribution.

### Public Domain Sources

US Government publications (NTSB, FAA, DOT) are in the **public domain** and may be freely used:

> "A work of the United States government is not subject to copyright protection in the United States." — 17 U.S.C. § 105

---

## References

### Primary Standards

1. **NTSB. (2013).** *Aviation Occurrence Categories: Definitions and Usage Notes (Version 4.6)*. National Transportation Safety Board.
   - https://www.ntsb.gov/safety/data/Documents/datafiles/OccurrenceCategoryDefinitions.pdf

2. **Shappell, S.A. & Wiegmann, D.A. (2000).** *The Human Factors Analysis and Classification System (HFACS)*. DOT/FAA/AM-00/7. Federal Aviation Administration.
   - https://rosap.ntl.bts.gov/view/dot/21482

### Research Reports

3. **IATA. (2015).** *Loss of Control In-Flight Accident Analysis Report (1st Edition)*. International Air Transport Association.
   - https://flightsafety.org/wp-content/uploads/2017/07/IATA-LOC-I-1st-Ed-2015.pdf

4. **IATA. (2019).** *Loss of Control In-Flight Accident Analysis Report*. International Air Transport Association.
   - https://www.iata.org/contentassets/b6eb2adc248c484192101edd1ed36015/loc-i_2019.pdf

5. **IATA. (2018).** *Controlled Flight Into Terrain Accident Analysis Report (2008-2017 Data)*. International Air Transport Association.
   - https://www.iata.org/contentassets/06377898f60c46028a4dd38f13f979ad/cfit-report.pdf

### Regulatory Guidance

6. **EASA. (2024).** *Loss of Control (LOC-I)*. European Union Aviation Safety Agency.
   - https://www.easa.europa.eu/en/domains/general-aviation/flying-safely/loss-of-control

7. **FAA. (2022).** *Controlled Flight Into Terrain*. Federal Aviation Administration.
   - https://www.faa.gov/sites/faa.gov/files/2022-01/Controlled%20Flight%20into%20Terrain.pdf

### Knowledge Bases

8. **SKYbrary. (2024).** *CAST/ICAO Common Taxonomy Team (CICTT)*. SKYbrary Aviation Safety.
   - https://skybrary.aero/articles/casticao-common-taxonomy-team-cictt

9. **SKYbrary. (2024).** *Human Factors Analysis and Classification System (HFACS)*. SKYbrary Aviation Safety.
   - https://skybrary.aero/articles/human-factors-analysis-and-classification-system-hfacs

10. **SKYbrary. (2024).** *Controlled Flight Into Terrain (CFIT)*. SKYbrary Aviation Safety.
    - https://skybrary.aero/articles/controlled-flight-terrain-cfit

### Academic Texts

11. **Wiegmann, D.A. & Shappell, S.A. (2003).** *A Human Error Approach to Aviation Accident Analysis*. Ashgate Publishing.

---

*Report generated: January 2026*
*RiskRADAR Project - Aviation Safety Classification System*
