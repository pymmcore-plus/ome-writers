# OME-TIFF Technical Specification for Library Implementation

> [!NOTE]
> This document was written by Claude, using deep research mode and web fetch.

**Sources:**

- [OME-TIFF Specification
  (v6.3.1)](https://docs.openmicroscopy.org/ome-model/6.3.1/ome-tiff/specification.html)
- [OME-TIFF Specification
  (stable)](https://ome-model.readthedocs.io/en/stable/ome-tiff/specification.html)
- [Screen Plate Well
  Documentation](https://ome-model.readthedocs.io/en/stable/developers/screen-plate-well.html)
- [Compliant HCS
  Specification](https://ome-model.readthedocs.io/en/stable/specifications/compliant-hcs.html)
- [BigTIFF Design
  Specification](https://libtiff.gitlab.io/libtiff/specification/bigtiff.html)
- [OME Data Model
  PDF](https://ome-model.readthedocs.io/_/downloads/en/latest/pdf/)

---

The OME-TIFF format combines standard TIFF with embedded OME-XML metadata to
create self-describing microscopy files. This specification documents the
requirements for implementing a compliant OME-TIFF writer based on ome-model
version 6.3.1 and the 2016-06 XML schema. The keywords MUST, SHOULD, and MAY
follow RFC 2119 semantics.

## Core File Structure and Identification

An OME-TIFF file is a valid TIFF file (baseline or BigTIFF) with OME-XML
metadata embedded in the **ImageDescription tag (tag 270)** of the first IFD.
([spec](https://docs.openmicroscopy.org/ome-model/6.3.1/ome-tiff/specification.html))

The file extension MUST be `.ome.tif`, `.ome.tiff`, or for BigTIFF specifically
`.ome.tf2`, `.ome.tf8`, or `.ome.btf`.

Identification requires checking three elements:

1. Valid TIFF magic bytes (`II` or `MM` followed by version 42 for standard TIFF
   or 43 for BigTIFF)
2. Presence of tag 270 in the first IFD
3. OME namespace `http://www.openmicroscopy.org/Schemas/OME/2016-06` in the XML
   content

The OME-XML MUST be **UTF-8 encoded** without BOM. Writers SHOULD preface the
XML with this warning comment:

```xml
<!-- Warning: this comment is an OME-XML metadata block, which contains
crucial dimensional parameters and other important metadata. Please edit
cautiously (if at all), and back up the original data before doing so.
For more information, see the OME-TIFF documentation:
https://docs.openmicroscopy.org/latest/ome-model/ome-tiff/ -->
```

## IFD Arrangement Fundamentals

Each 2D image plane (XY slice) occupies exactly **one IFD**. The total number of
IFDs for a single Image equals `SizeZ × SizeC × SizeT`. IFD tag entries MUST be
sorted in ascending numerical order per TIFF 6.0 specification. The OME-XML
metadata block MUST appear only in the first IFD's ImageDescription
tag—subsequent IFDs MAY have empty or absent tag 270.
([spec](https://ome-model.readthedocs.io/en/stable/ome-tiff/specification.html))

The **DimensionOrder** attribute controls IFD rasterization. The six valid
values specify which dimension varies fastest after X and Y:

| DimensionOrder | Index Formula | Iteration Order |
|----------------|---------------|-----------------|
| XYZCT | `z + (c × SizeZ) + (t × SizeZ × SizeC)` | Z→C→T |
| XYZTC | `z + (t × SizeZ) + (c × SizeZ × SizeT)` | Z→T→C |
| XYCZT | `c + (z × SizeC) + (t × SizeC × SizeZ)` | C→Z→T |
| XYCTZ | `c + (t × SizeC) + (z × SizeC × SizeT)` | C→T→Z |
| XYTCZ | `t + (c × SizeT) + (z × SizeT × SizeC)` | T→C→Z |
| XYTZC | `t + (z × SizeT) + (c × SizeT × SizeZ)` | T→Z→C |

For example, with SizeZ=10, SizeC=3, SizeT=5 and DimensionOrder=XYZCT, the plane
at Z=4, C=1, T=2 maps to IFD index `4 + (1×10) + (2×10×3) = 74`.

## BigTIFF Requirements

Writers MUST use BigTIFF format when file size exceeds **4GB**. ([BigTIFF
spec](https://libtiff.gitlab.io/libtiff/specification/bigtiff.html))

BigTIFF uses:

- Version number 43 (vs 42)
- 8-byte offsets throughout
- 8-byte IFD entry counts
- 20-byte IFD entries (vs 12)

The format introduces three new data types: `LONG8` (type 16), `SLONG8` (type
17), and `IFD8` (type 18). Writers SHOULD proactively use BigTIFF for large
pyramidal images even when initially under 4GB, as the final size may exceed
this limit.

## Pixel Type Mappings

The Pixels `Type` attribute maps to TIFF tags as follows:

| OME Type | BitsPerSample | SampleFormat |
|----------|---------------|--------------|
| uint8 | 8 | 1 (UINT) |
| uint16 | 16 | 1 (UINT) |
| uint32 | 32 | 1 (UINT) |
| int8 | 8 | 2 (INT) |
| int16 | 16 | 2 (INT) |
| int32 | 32 | 2 (INT) |
| float | 32 | 3 (IEEEFP) |
| double | 64 | 3 (IEEEFP) |
| complex | 64 | 6 (COMPLEXIEEEFP) |
| double-complex | 128 | 6 (COMPLEXIEEEFP) |

Both little-endian (`II`) and big-endian (`MM`) byte orders are valid. The
Pixels `BigEndian` attribute SHOULD match the actual TIFF byte order.

## Compression Options

OME-TIFF files MAY use any standard TIFF compression:

- **Uncompressed** (default)
- **LZW** (lossless, widely compatible)
- **JPEG** (lossy, 8-bit only)
- **JPEG-2000** (lossy or lossless)

Large images SHOULD use tiles rather than strips—typical tile sizes are
**256×256** or **512×512** pixels.

---

## OME-XML Schema Structure

The root OME element requires the namespace declaration and MAY include a UUID:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 
     http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
     UUID="urn:uuid:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx">
```

### Image Element

Each Image represents one 5D dataset (series). Required attributes:

- **ID**: Unique identifier in format `Image:n` (e.g., `Image:0`)

Optional but recommended: `Name` for human-readable identification.

### Pixels Element (required child of Image)

All these attributes are **REQUIRED**:

| Attribute | Type | Description |
|-----------|------|-------------|
| ID | string | Unique identifier (`Pixels:n`) |
| DimensionOrder | enum | One of the six valid values |
| Type | enum | Pixel data type |
| SizeX, SizeY | positiveInt | Plane dimensions |
| SizeZ, SizeC, SizeT | positiveInt | Stack dimensions |

Optional attributes include `BigEndian`, `Interleaved`, `SignificantBits`, and
physical size attributes (`PhysicalSizeX/Y/Z` with corresponding unit
attributes, defaulting to micrometers).

### Channel Element

Channel elements are children of Pixels. Writers SHOULD include exactly `SizeC`
Channel elements. Each MUST have a unique `ID` attribute. Optional attributes
include `Name`, `Color` (RGBA integer), `SamplesPerPixel` (1 for grayscale, 3
for RGB), and wavelength information.

### TiffData Element (critical for OME-TIFF)

TiffData elements map IFDs to dimensional positions. At minimum, one TiffData
element is REQUIRED:

| Attribute | Default | Description |
|-----------|---------|-------------|
| IFD | 0 | Starting IFD index (0-based) |
| FirstZ | 0 | Z position at this IFD |
| FirstC | 0 | C position at this IFD |
| FirstT | 0 | T position at this IFD |
| PlaneCount | all/1 | Number of consecutive IFDs |

The simplest usage assigns all planes automatically:

```xml
<TiffData/>
```

This maps IFDs 0 through (SizeZ×SizeC×SizeT−1) according to DimensionOrder. For
explicit control:

```xml
<TiffData IFD="100" PlaneCount="50"/>
```

---

## Use Case 1: Single 5D Image (TCZYX)

### IFD Structure

For a single Image with SizeZ=10, SizeC=3, SizeT=5, the file contains **150
IFDs** (10×3×5) numbered 0-149. With DimensionOrder=XYZCT, Z varies fastest:

- IFDs 0-9: Z0-9, C0, T0
- IFDs 10-19: Z0-9, C1, T0
- IFDs 20-29: Z0-9, C2, T0
- IFDs 30-39: Z0-9, C0, T1
- ...and so on

### Multi-Resolution Pyramid Support

OME-TIFF officially supports pyramidal images as of **ome-model 6.0.0**.
([spec](https://docs.openmicroscopy.org/ome-model/6.3.1/ome-tiff/specification.html))

**MUST requirements:**

- Sub-resolution IFDs MUST NOT appear in the main IFD chain
- Sub-resolution IFDs MUST NOT be referenced in TiffData elements
- Full-resolution IFDs MUST reference sub-resolution levels using **SubIFDs tag
  (tag 330)**
- SubIFD offsets MUST be ordered from largest to smallest resolution

**SHOULD requirements:**

- NewSubFileType tag (254) SHOULD be set to 1 for sub-resolution planes
- Downsampling factor SHOULD be a power of 2 (typically 2)
- Factor SHOULD be identical for X and Y dimensions
- Factor SHOULD be consistent between consecutive levels

The pyramid structure applies **per full-resolution plane**—each Z/C/T
combination has its own SubIFDs chain. Pyramids only downsample X and Y; Z, C,
and T dimensions remain unchanged. The OME-XML contains **no Resolution
element**; readers discover pyramid levels by reading SubIFDs from each
full-resolution IFD.

```
Main IFD Chain (full-resolution):
IFD0 → IFD1 → IFD2 → ... → IFD149 → NULL
  |       |
  ↓       ↓
SubIFDs  SubIFDs
  |       |
  ├─ Level1 (2048×2048)
  ├─ Level2 (1024×1024)
  └─ Level3 (512×512)
```

### Minimum Valid XML for 5D Image

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 
     http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
  <Image ID="Image:0" Name="5D Stack">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
            SizeX="4096" SizeY="4096" SizeZ="10" SizeC="3" SizeT="5"
            PhysicalSizeX="0.65" PhysicalSizeY="0.65" PhysicalSizeZ="2.0">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:2" SamplesPerPixel="1"/>
      <TiffData/>
    </Pixels>
  </Image>
</OME>
```

---

## Use Case 2: Multiple Positions (PTCZYX)

### Series Concept

In OME-TIFF, each **Image element equals one series**. A file MAY contain
multiple Image elements, allowing multiple positions in a single file. Each
Image has independent dimensions, pixel type, and TiffData mappings.
([spec](https://ome-model.readthedocs.io/en/stable/ome-tiff/specification.html))

### Single-File Approach

For 3 positions, each with Z=10, C=2, T=5 (100 planes each), the file contains
**300 total IFDs**:

```xml
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     UUID="urn:uuid:abc12345-...">
  <Image ID="Image:0" Name="Position 1">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
            SizeX="512" SizeY="512" SizeZ="10" SizeC="2" SizeT="5">
      <Channel ID="Channel:0:0"/><Channel ID="Channel:0:1"/>
      <TiffData IFD="0" PlaneCount="100"/>
    </Pixels>
  </Image>
  <Image ID="Image:1" Name="Position 2">
    <Pixels ID="Pixels:1" DimensionOrder="XYZCT" Type="uint16"
            SizeX="512" SizeY="512" SizeZ="10" SizeC="2" SizeT="5">
      <Channel ID="Channel:1:0"/><Channel ID="Channel:1:1"/>
      <TiffData IFD="100" PlaneCount="100"/>
    </Pixels>
  </Image>
  <Image ID="Image:2" Name="Position 3">
    <Pixels ID="Pixels:2" DimensionOrder="XYZCT" Type="uint16"
            SizeX="512" SizeY="512" SizeZ="10" SizeC="2" SizeT="5">
      <Channel ID="Channel:2:0"/><Channel ID="Channel:2:1"/>
      <TiffData IFD="200" PlaneCount="100"/>
    </Pixels>
  </Image>
</OME>
```

### Multi-File Approach

When distributing across multiple files, each file MUST have a **distinct UUID**
on the root OME element. TiffData elements reference external files using the
UUID child element:

```xml
<TiffData FirstC="0" FirstT="0" FirstZ="0" IFD="0" PlaneCount="100">
  <UUID FileName="position_1.ome.tiff">urn:uuid:aaa12345-...</UUID>
</TiffData>
```

**UUID requirements:**

- Format: `urn:uuid:` followed by standard UUID (8-4-4-4-12 hex digits)
- Root OME UUID MUST be unique per file
- TiffData UUID content MUST match the target file's root OME UUID
- FileName attribute is optional but STRONGLY RECOMMENDED

**Metadata duplication options:**

1. **Full embedded (recommended)**: Each file contains complete OME-XML; only
   root UUID differs. Provides redundancy.

2. **BinaryOnly with companion**: Secondary files contain minimal XML pointing
   to a master:

```xml
<OME UUID="urn:uuid:secondary-file-uuid">
  <BinaryOnly MetadataFile="master.companion.ome"
              UUID="urn:uuid:master-file-uuid"/>
</OME>
```

The companion file (`.companion.ome`) contains the complete OME-XML without
pixel data.

---

## Use Case 3: HCS Plate Data

### SPW Model Overview

High Content Screening uses the Screen-Plate-Well (SPW) model. ([SPW
docs](https://ome-model.readthedocs.io/en/stable/developers/screen-plate-well.html))

The hierarchy is: **Plate → Well → WellSample → ImageRef → Image**. Plate and
Screen are top-level siblings (not parent-child), allowing plates to belong to
multiple screens.

### Plate Element

Required attribute: **ID** (e.g., `Plate:1`)

Recommended attributes:

| Attribute | Description |
|-----------|-------------|
| Name | Human-readable identifier |
| Rows, Columns | Plate dimensions |
| RowNamingConvention | "letter" or "number" |
| ColumnNamingConvention | "letter" or "number" |

### Well Element

Required attributes:

- **ID**: Unique identifier (`Well:n`)
- **Row**: Row index (0-based, origin top-left)
- **Column**: Column index (0-based)

Well A1 on a 96-well plate has Row="0", Column="0".

### WellSample Element (critical link)

WellSample connects plate structure to image data. Each field of view requires a
separate WellSample. ([HCS
spec](https://ome-model.readthedocs.io/en/stable/specifications/compliant-hcs.html))

Required attributes:

- **ID**: Unique identifier
- **Index**: MUST be unique within the Plate (need not be sequential)

Recommended attributes:

- **PositionX, PositionY**: Field position relative to well origin

The ImageRef child links to the actual image data:

```xml
<WellSample ID="WellSample:0" Index="0" PositionX="0" PositionY="0">
  <ImageRef ID="Image:0"/>
</WellSample>
```

### Mapping Principle

**One WellSample = One Image element**. For a 96-well plate with 4 fields per
well:

- 96 wells × 4 fields = **384 WellSample elements**
- 384 corresponding Image elements
- Each Image may have multiple IFDs (for Z-stacks, channels, timepoints)

### Complete HCS XML Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Plate ID="Plate:1" Name="Experiment 1" 
         Rows="8" Columns="12"
         RowNamingConvention="letter" 
         ColumnNamingConvention="number">
    
    <Well ID="Well:0" Row="0" Column="0">
      <WellSample ID="WellSample:0" Index="0">
        <ImageRef ID="Image:0"/>
      </WellSample>
      <WellSample ID="WellSample:1" Index="1">
        <ImageRef ID="Image:1"/>
      </WellSample>
    </Well>
    
    <Well ID="Well:1" Row="0" Column="1">
      <WellSample ID="WellSample:2" Index="2">
        <ImageRef ID="Image:2"/>
      </WellSample>
      <WellSample ID="WellSample:3" Index="3">
        <ImageRef ID="Image:3"/>
      </WellSample>
    </Well>
    <!-- Additional wells... -->
  </Plate>

  <Image ID="Image:0" Name="A1-Field1">
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16"
            SizeX="1024" SizeY="1024" SizeC="3" SizeZ="1" SizeT="1">
      <Channel ID="Channel:0:0"/>
      <Channel ID="Channel:0:1"/>
      <Channel ID="Channel:0:2"/>
      <TiffData/>
    </Pixels>
  </Image>
  <!-- Additional Image elements... -->
</OME>
```

### Single vs Multi-File for HCS

A complete plate CAN fit in one BigTIFF file, but multi-file organization is
common:

1. **One file per field**: `plate_A01_field1.ome.tiff`,
   `plate_A01_field2.ome.tiff`
2. **One file per well**: `plate_A01.ome.tiff` containing all fields
3. **Companion file**: Multiple `.ome.tiff` files with BinaryOnly metadata plus
   one `.companion.ome` with complete plate structure

### Optional HCS Elements

**PlateAcquisition**: Records acquisition metadata (StartTime, EndTime,
MaximumFieldCount). Useful for time-course experiments with multiple passes.
Optional for simple acquisitions.

**Screen**: Groups plates into experimental sets, defines reagents. Not required
for valid HCS data—use when modeling actual screening workflows with reagent
tracking.

---

## Consolidated Requirements Summary

### File Structure MUST

- Store UTF-8 encoded OME-XML in tag 270 of first IFD only
- Create exactly one IFD per 2D plane in main chain
- Sort IFD tag entries in ascending numerical order
- Use BigTIFF when file size exceeds 4GB
- Use SubIFDs tag (330) for pyramid levels, never main IFD chain
- Include distinct UUID on root OME element for each file in multi-file sets
- Ensure WellSample Index values are unique within a Plate

### File Structure SHOULD

- Use `.ome.tif` or `.ome.tiff` extension
- Include warning comment before OME-XML
- Use tiles (not strips) for images larger than 4096×4096
- Set NewSubFileType=1 for pyramid sub-resolutions
- Use power-of-two downsampling for pyramids
- Include UUID FileName attribute in TiffData external references
- Embed full OME-XML in each file of multi-file datasets

### File Structure MAY

- Use any supported compression (uncompressed, LZW, JPEG, JPEG-2000)
- Use companion OME-XML files with BinaryOnly references
- Include Screen and PlateAcquisition elements for HCS
- Use explicit TiffData mappings to override DimensionOrder defaults
- Include physical size and timing metadata

---

## Implementation Checklist

A compliant OME-TIFF writer must implement:

1. **TIFF layer**: Valid TIFF/BigTIFF with properly sorted IFD entries and
   correct pixel type tags
2. **XML generation**: Schema-valid OME-XML with required elements and
   attributes
3. **TiffData mapping**: Correct correspondence between IFD indices and
   dimensional coordinates
4. **Pyramid support**: SubIFDs structure with proper NewSubFileType tagging
5. **Multi-file support**: UUID generation and TiffData external references
6. **HCS support**: Plate/Well/WellSample hierarchy with proper ImageRef linkage

Validation against the official XSD schema
(`http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd`) is strongly
recommended before writing.

---

## References

1. OME-TIFF Specification (v6.3.1):
   <https://docs.openmicroscopy.org/ome-model/6.3.1/ome-tiff/specification.html>
2. OME-TIFF Specification (stable):
   <https://ome-model.readthedocs.io/en/stable/ome-tiff/specification.html>
3. Screen Plate Well Documentation:
   <https://ome-model.readthedocs.io/en/stable/developers/screen-plate-well.html>
4. Compliant HCS Specification:
   <https://ome-model.readthedocs.io/en/stable/specifications/compliant-hcs.html>
5. BigTIFF Design: <https://libtiff.gitlab.io/libtiff/specification/bigtiff.html>
6. OME Data Model PDF:
   <https://ome-model.readthedocs.io/_/downloads/en/latest/pdf/>
7. OME XML Schema: <http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd>
