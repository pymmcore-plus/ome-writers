---
title: Event-Driven Example
---
# Non-Deterministic, Event-Driven Imaging

This is our current recommendation where the higher-order structure of the dataset is
not known in advance.  In this pattern, we assume only that a series of 2D frames
will be acquired, and that each frame has some associated metadata.  This is the most
general case (though may lack convenience features of more structured sub-cases).

```python
--8<-- "examples/event_driven.py"
```

Example metadata output for OME-TIFF and OME-Zarr are shown below:

=== "OME-TIFF Metadata"

    Special keys `"delta_t"`, `"exposure_time"`, `"position_x"`, `"position_y"`,
    and `"position_z"` will be written as Plane attributes, while any additional
    metadata (in this case `"temperature"` and `"laser_power"`) will be written
    as MapAnnotations referenced by each Plane:

    ```xml title="OME-XML (in Tiff Header or .companion.ome file)"
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06
                             http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
         UUID="urn:uuid:7c3ba826-76d9-49c4-933f-96bb2a0bce74">
      <Image ID="Image:0" Name="0">
        <AcquisitionDate>2026-01-27T02:33:05.778432Z</AcquisitionDate>
        <Pixels ID="Pixels:0" DimensionOrder="XYTCZ" Type="uint16"
                SizeX="256" SizeY="256" SizeZ="1" SizeC="1" SizeT="5">
          <Channel ID="Channel:0:0"/>
          <TiffData IFD="0" PlaneCount="5"/>
          <Plane TheZ="0" TheT="0" TheC="0" DeltaT="3E-06" ExposureTime="0.01"
                 PositionX="100.0" PositionY="200.0" PositionZ="50.0"> <!-- (1)! -->
            <AnnotationRef ID="Annotation:0"/> <!-- (2)! -->
          </Plane>
          <Plane TheZ="0" TheT="1" TheC="0" DeltaT="0.000363" ExposureTime="0.01"
                 PositionX="100.5" PositionY="200.3" PositionZ="50.0">
            <AnnotationRef ID="Annotation:1"/>
          </Plane>
          <Plane TheZ="0" TheT="2" TheC="0" DeltaT="0.000573" ExposureTime="0.01"
                 PositionX="101.0" PositionY="200.6" PositionZ="50.0">
            <AnnotationRef ID="Annotation:2"/>
          </Plane>
          <Plane TheZ="0" TheT="3" TheC="0" DeltaT="0.000679" ExposureTime="0.01"
                 PositionX="101.5" PositionY="200.9" PositionZ="50.0">
            <AnnotationRef ID="Annotation:3"/>
          </Plane>
          <Plane TheZ="0" TheT="4" TheC="0" DeltaT="0.000802" ExposureTime="0.01"
                 PositionX="102.0" PositionY="201.2" PositionZ="50.0">
            <AnnotationRef ID="Annotation:4"/>
          </Plane>
        </Pixels>
      </Image>
      <StructuredAnnotations> <!-- (3)! -->
        <MapAnnotation ID="Annotation:0">
          <Value>
            <M K="temperature">37.176405234596764</M>
            <M K="laser_power">50.0</M>
          </Value>
        </MapAnnotation>
        <MapAnnotation ID="Annotation:1">
          <Value>
            <M K="temperature">37.04001572083672</M>
            <M K="laser_power">50.0</M>
          </Value>
        </MapAnnotation>
        <MapAnnotation ID="Annotation:2">
          <Value>
            <M K="temperature">36.831773331171895</M>
            <M K="laser_power">50.0</M>
          </Value>
        </MapAnnotation>
        <MapAnnotation ID="Annotation:3">
          <Value>
            <M K="temperature">36.904821272767585</M>
            <M K="laser_power">50.0</M>
          </Value>
        </MapAnnotation>
        <MapAnnotation ID="Annotation:4">
          <Value>
            <M K="temperature">36.696539103178715</M>
            <M K="laser_power">50.0</M>
          </Value>
        </MapAnnotation>
      </StructuredAnnotations>
    </OME>
    ```

    1.  :eyes: Special keys `delta_t`, `exposure_time`, `position_x`, `position_y`,
        and `position_z` are written as Plane attributes
    2.  :eyes: All other key/value pairs are gathered into a `MapAnnotation`. That
        `MapAnnotation` is stored in `<StructuredAnnotations>`, and a reference is 
        attached to the specific `<Plane>`.
    3.  All non-standard data (for all planes, across all images) are gathered as
        `MapAnnotation`s in the global `<StructuredAnnotations>` section (as per the
        OME-XML specification).

=== "OME-Zarr Metadata"

    If frame metadata is provided, the `attributes` dict each multiscales image group
    will include a `ome_writers` section (sibling to `"ome"`), with a `frame_metadata`
    key containing a list of per-frame metadata dictionaries.  `storage_index` indicates
    the exact index of the frame in the Zarr array:

    ```json title="root/zarr.json"
    {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": { ... },
            "ome_writers": {  // (1)!
                "version": "0.1.0rc2.dev5+g34cbf69ed.d20260127",
                "frame_metadata": [ // (2)!
                    {
                        "delta_t": 7e-06,
                        "exposure_time": 0.01,
                        "position_x": 100.0,
                        "position_y": 200.0,
                        "position_z": 50.0,
                        "temperature": 37.176405234596764,
                        "laser_power": 50.0,
                        "storage_index": [
                            0
                        ]
                    },
                    {
                        "delta_t": 0.000657,
                        "exposure_time": 0.01,
                        "position_x": 100.5,
                        "position_y": 200.3,
                        "position_z": 50.0,
                        "temperature": 37.04001572083672,
                        "laser_power": 50.0,
                        "storage_index": [
                            1
                        ]
                    },
                    {
                        "delta_t": 0.003701,
                        "exposure_time": 0.01,
                        "position_x": 101.0,
                        "position_y": 200.6,
                        "position_z": 50.0,
                        "temperature": 36.831773331171895,
                        "laser_power": 50.0,
                        "storage_index": [
                            2
                        ]
                    },
                    {
                        "delta_t": 0.007688,
                        "exposure_time": 0.01,
                        "position_x": 101.5,
                        "position_y": 200.9,
                        "position_z": 50.0,
                        "temperature": 36.904821272767585,
                        "laser_power": 50.0,
                        "storage_index": [
                            3
                        ]
                    },
                    {
                        "delta_t": 0.010254,
                        "exposure_time": 0.01,
                        "position_x": 102.0,
                        "position_y": 201.2,
                        "position_z": 50.0,
                        "temperature": 36.696539103178715,
                        "laser_power": 50.0,
                        "storage_index": [
                            4
                        ]
                    }
                ]
            }
        }
    }
    ```

    1. :eyes: The `ome_writers` section contains all data that does not yet
       have a standard location in the OME-Zarr specification.
    2. :eyes: `frame_metadata` appended during the stream are gathered in the
       `"frame_metadata"` list, with each entry containing the metadata for a specific frame,
       along with its `storage_index` in the associated Zarr array.
