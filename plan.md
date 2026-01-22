# TIFF Stream Updates

I'd like to update the current `TiffBackend` implementation for writing OME-TIFF files with 2 main changes:
1. add more detailed OME-XML metadata, including Plane and TiffData elements for each dimension in the 5D series (c, z, t) (see **Single Position** below).
2. update the metadata for the multiposition case to store the full OME-XML in each OME-TIFF file for each position, with each file referenced through its UUID in the metadata

## Single Position

### 2 channel

**Current**:

```xml
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
  <Image ID="Image:0" Name="example_5d_image.ome">
    <Pixels ID="Pixels:0" DimensionOrder="XYCTZ" Type="uint16" BigEndian="false" SizeX="256" SizeY="256" SizeZ="1" SizeC="2" SizeT="1">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" SamplesPerPixel="1"/>
      <TiffData PlaneCount="2"/>
    </Pixels>
  </Image>
</OME>
```

**Updated**:

```xml
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
  <Image ID="Image:0" Name="example_5d_image">
    <AcquisitionDate>2026-01-22T14:23:43.392475-05:00</AcquisitionDate>
    <Pixels ID="Pixels:0" DimensionOrder="XYCTZ" Type="uint16" BigEndian="false" SizeX="256" SizeY="256" SizeZ="1" SizeC="2" SizeT="1" PhysicalSizeX="0.5" PhysicalSizeXUnit="µm" PhysicalSizeY="0.5" PhysicalSizeYUnit="µm">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" SamplesPerPixel="1"/>
      <TiffData PlaneCount="2"/>
    </Pixels>
  </Image>
</OME>
```

### 2 channel, 3 slices

**Current**:

```xml
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
  <Image ID="Image:0" Name="example_5d_image.ome">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" BigEndian="false" SizeX="256" SizeY="256" SizeZ="3" SizeC="2" SizeT="1">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" SamplesPerPixel="1"/>
      <TiffData PlaneCount="6"/>
    </Pixels>
  </Image>
</OME>
```

**Updated**:

```xml
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
  <Image ID="Image:0" Name="example_5d_image">
    <AcquisitionDate>2026-01-22T14:33:41.971494-05:00</AcquisitionDate>
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" SizeX="256" SizeY="256" SizeZ="3" SizeC="2" SizeT="1" PhysicalSizeX="0.5" PhysicalSizeXUnit="µm" PhysicalSizeY="0.5" PhysicalSizeYUnit="µm" PhysicalSizeZ="1.0" PhysicalSizeZUnit="µm">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" SamplesPerPixel="1"/>
      <TiffData PlaneCount="6"/>
    </Pixels>
  </Image>
</OME>
```

## Multiposition

### 2 channel, 3, slices, 2 positions

Multiple OME-TIFF files, one per position.

**Current**:

File p000:

```xml
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
  <Image ID="Image:0" Name="example_5d_series_p000.ome">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" BigEndian="false" SizeX="256" SizeY="256" SizeZ="3" SizeC="2" SizeT="1">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" SamplesPerPixel="1"/>
      <TiffData PlaneCount="6"/>
    </Pixels>
  </Image>
</OME>
```

File p001:

```xml
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
  <Image ID="Image:1" Name="example_5d_series_p001.ome">
    <Pixels ID="Pixels:1" DimensionOrder="XYZCT" Type="uint16" BigEndian="false" SizeX="256" SizeY="256" SizeZ="3" SizeC="2" SizeT="1">
      <Channel ID="Channel:1:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:1:1" SamplesPerPixel="1"/>
      <TiffData PlaneCount="6"/>
    </Pixels>
  </Image>
</OME>
```

**Updated**:

In this example we only have 2 positions each with 2 channels and 3 slices, but the same idea applies if we have also t. We will simply have TiffData and Plane elements for each dimension in the 5D series.

The full OME-XML stored in each OME-TIFF file for each position. In the metadata, each file is referenced through its UUID.

```xml
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 
http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd" Creator="ome_writers v0.0.1b2.dev20+gd4e44f5e0.d20260122">
  <Image ID="Image:0" Name="example_5d_series_p000">
    <AcquisitionDate>2026-01-22T14:33:41.971494-05:00</AcquisitionDate>  # added AcquisitionDate element
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint8" SizeX="256" SizeY="256" SizeZ="3" SizeC="2" SizeT="1" PhysicalSizeX="0.5" PhysicalSizeXUnit="µm" PhysicalSizeY="0.5" PhysicalSizeYUnit="µm" PhysicalSizeZ="1.0" PhysicalSizeZUnit="µm" >
      <Channel ID="Channel:1:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:1:1" SamplesPerPixel="1"/>
      <TiffData IFD="0" FirstZ="0" FirstT="0" FirstC="0" PlaneCount="1">
        <UUID FileName="tiff_example_p000.ome.tiff">urn:uuid:b7f10cb1-636c-474c-8501-eda9f0c44665</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="1" FirstZ="0" FirstT="0" FirstC="1" PlaneCount="1">
        <UUID FileName="tiff_example_p000.ome.tiff">urn:uuid:b7f10cb1-636c-474c-8501-eda9f0c44665</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="2" FirstZ="1" FirstT="0" FirstC="0" PlaneCount="1">
        <UUID FileName="tiff_example_p000.ome.tiff">urn:uuid:b7f10cb1-636c-474c-8501-eda9f0c44665</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="3" FirstZ="1" FirstT="0" FirstC="1" PlaneCount="1">
        <UUID FileName="tiff_example_p000.ome.tiff">urn:uuid:b7f10cb1-636c-474c-8501-eda9f0c44665</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="4" FirstZ="2" FirstT="0" FirstC="0" PlaneCount="1">
        <UUID FileName="tiff_example_p000.ome.tiff">urn:uuid:b7f10cb1-636c-474c-8501-eda9f0c44665</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="5" FirstZ="2" FirstT="0" FirstC="1" PlaneCount="1">
        <UUID FileName="tiff_example_p000.ome.tiff">urn:uuid:b7f10cb1-636c-474c-8501-eda9f0c44665</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
    </Pixels>
  </Image>
  <Image ID="Image:1" Name="example_5d_series_p001">
  <AcquisitionDate>2026-01-22T14:35:41.971494-05:00</AcquisitionDate>  # added AcquisitionDate element
    <Pixels ID="Pixels:1" DimensionOrder="XYCZT" Type="uint8" SizeX="256" SizeY="256" SizeZ="3" SizeC="2" SizeT="1" PhysicalSizeX="0.5" PhysicalSizeXUnit="µm" PhysicalSizeY="0.5" PhysicalSizeYUnit="µm" PhysicalSizeZ="1.0" PhysicalSizeZUnit="µm">
      <Channel ID="Channel:1:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:1:1" SamplesPerPixel="1"/>
      <TiffData IFD="0" FirstZ="0" FirstT="0" FirstC="0" PlaneCount="1">
        <UUID FileName="tiff_example_p001.ome.tiff">urn:uuid:abce0a63-96ba-4963-ad6f-71037a45a38a</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="1" FirstZ="0" FirstT="0" FirstC="1" PlaneCount="1">
        <UUID FileName="tiff_example_p001.ome.tiff">urn:uuid:abce0a63-96ba-4963-ad6f-71037a45a38a</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="2" FirstZ="1" FirstT="0" FirstC="0" PlaneCount="1">
        <UUID FileName="tiff_example_p001.ome.tiff">urn:uuid:abce0a63-96ba-4963-ad6f-71037a45a38a</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="3" FirstZ="1" FirstT="0" FirstC="1" PlaneCount="1">
        <UUID FileName="tiff_example_p001.ome.tiff">urn:uuid:abce0a63-96ba-4963-ad6f-71037a45a38a</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="4" FirstZ="2" FirstT="0" FirstC="0" PlaneCount="1">
        <UUID FileName="tiff_example_p001.ome.tiff">urn:uuid:abce0a63-96ba-4963-ad6f-71037a45a38a</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
      <TiffData IFD="5" FirstZ="2" FirstT="0" FirstC="1" PlaneCount="1">
        <UUID FileName="tiff_example_p001.ome.tiff">urn:uuid:abce0a63-96ba-4963-ad6f-71037a45a38a</UUID>  # added UUID element referencing file through UUID (and name)
      </TiffData>
    </Pixels>
  </Image>
</OME>
```
