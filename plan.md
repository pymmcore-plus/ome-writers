
# Single Position

## 2 channel

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
UUID="urn:uuid:c52210d1-b8e3-49dd-8bcb-2f4ad0eda034">  # added UUID attribute
  <Image ID="Image:0" Name="example_5d_image">  # removed .ome from Name attribute
    <AcquisitionDate>2026-01-22T14:23:43.392475-05:00</AcquisitionDate>  # added AcquisitionDate element
    <Pixels ID="Pixels:0" DimensionOrder="XYCTZ" Type="uint16" BigEndian="false" SizeX="256" SizeY="256" SizeZ="1" SizeC="2" SizeT="1">  # pymm should add PhysicalSizeX/Y and Unit attributes
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>  # pymm should add channel name
      <Channel ID="Channel:0:1" SamplesPerPixel="1"/>  # pymm should add channel name
      <TiffData FirstZ="0" FirstT="0" FirstC="0" PlaneCount="1"/>  # added TiffData elements
      <TiffData FirstZ="0" FirstT="0" FirstC="1" PlaneCount="1"/>  # added TiffData elements
      <Plane TheZ="0" TheT="0" TheC="0"/>  # added Plane elements, pymm should add delta, exposure, position, etc info
      <Plane TheZ="0" TheT="0" TheC="1"/>  # added Plane elements, pymm should add delta, exposure, position, etc info
    </Pixels>
  </Image>
</OME>
```

## 2 channel, 3 slices

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
UUID="urn:uuid:9d6001f0-8ef6-4b45-a662-b38d1025514a">  # added UUID attribute
  <Image ID="Image:0" Name="example_5d_image">  # removed .ome from Name attribute
    <AcquisitionDate>2026-01-22T14:33:41.971494-05:00</AcquisitionDate>  # added AcquisitionDate element
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" SizeX="100" SizeY="200" SizeZ="3" SizeC="2" SizeT="1"> # pymm should add PhysicalSizeX/Y and Unit attributes
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>  # pymm should add channel name
      <Channel ID="Channel:0:1" SamplesPerPixel="1"/>  # pymm should add channel name
      <TiffData FirstZ="0" FirstT="0" FirstC="0" PlaneCount="1"/>  # added TiffData elements
      <TiffData FirstZ="1" FirstT="0" FirstC="0" PlaneCount="1"/>  # added TiffData elements
      <TiffData FirstZ="2" FirstT="0" FirstC="0" PlaneCount="1"/>  # added TiffData elements
      <TiffData FirstZ="0" FirstT="0" FirstC="1" PlaneCount="1"/>  # added TiffData elements
      <TiffData FirstZ="1" FirstT="0" FirstC="1" PlaneCount="1"/>  # added TiffData elements
      <TiffData FirstZ="2" FirstT="0" FirstC="1" PlaneCount="1"/>  # added TiffData elements
      <Plane TheZ="0" TheT="0" TheC="0"/>  # added Plane elements, pymm should add delta, exposure, position, etc info
      <Plane TheZ="1" TheT="0" TheC="0"/>  # added Plane elements, pymm should add delta, exposure, position, etc info
      <Plane TheZ="2" TheT="0" TheC="0"/>  # added Plane elements, pymm should add delta, exposure, position, etc info
      <Plane TheZ="0" TheT="0" TheC="1"/>  # added Plane elements, pymm should add delta, exposure, position, etc info
      <Plane TheZ="1" TheT="0" TheC="1"/>  # added Plane elements, pymm should add delta, exposure, position, etc info
      <Plane TheZ="2" TheT="0" TheC="1"/>  # added Plane elements, pymm should add delta, exposure, position, etc info
    </Pixels>
  </Image>
</OME>
```

## 2 channel, 2 positions

**Current**:

```xml
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
  <Image ID="Image:0" Name="example_5d_series_p000.ome">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" BigEndian="false" SizeX="256" SizeY="256" SizeZ="4" SizeC="3" SizeT="2">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:2" SamplesPerPixel="1"/>
      <TiffData PlaneCount="24"/>
    </Pixels>
  </Image>
</OME>
```

**Updated**:

```xml

```
