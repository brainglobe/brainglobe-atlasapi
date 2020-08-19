---
title: 'brainglobe atlas API: a common interface for neuroanatomical atlases'
tags:
  - Python
  - neuroscience
  - neuroanatomy
  - microscopy

authors:
  - name: Federico Claudi^[Joint first author, ordered alphabetically]
    affiliation: 1
  - name: Luigi Petrucco^[Joint first author, ordered alphabetically]
    affiliation: 2
  - name: Adam Tyson^[Joint first author, ordered alphabetically]
    orcid: 0000-0003-3225-1130
    affiliation: 1
  - name: Tiago Branco^[Joint senior author, ordered alphabetically]
    affiliation: 1
  - name: Troy Margrie^[Joint senior author, ordered alphabetically]
    affiliation: 1
  - name: Ruben Portugues^[Joint senior author, ordered alphabetically]
    affiliation: "2, 3"

affiliations:
 - name: Sainsbury Wellcome Centre, University College London. London, U.K.
   index: 1
 - name: Max Planck Institute of Neurobiology. Munich, Germany
   index: 2
 - name:  Institute of Neuroscience, Technical University of Munich. Munich, Germany
   index: 3
date: 19 August 2020
bibliography: paper.bib

---

# Summary
Neuroanatomical data analysis relies on precise understanding of where in the brian the information comes from. For this reason, brain atlases have been developed which provide brain region annotation overlaid upon a standardized reference brain image. These atlases have been developed for many animal model species, and are routinely used for data analysis and visualisation. The availability of these atlases vary, many are available online and some have an application programming interface (API), but these are inconsistent. This hinders the development and adoption of open-source neuroanatomy software, as each tool is typically only developed for a single atlas for one model organism. The brainglobe atlas API (BG-Atlas API) overcomes this problem by providing a common interface for programmers to download and process data from multiple atlases. Software can then be developed agnostic to the atlas used, increasing adoption and interopability of software in neuroscience. 

# Statement of need 
Neuroscience is gradually becoming a field reliant on large scale data acquisition, such as whole-brain optical neural recordings [@Ahrens:2012], brain-wide connectivity mapping [@Osten:2013] or high-density electrophysiological probes [@Jun:2017]. For this reason, custom, computational analysis is now necessary in the majority of neuroscience laboratories, and Python is gradually emerging as the language of choice [@Muller:2015].

The brain of even the simplest organism is a highly complex structure. In mammals and other vertebrates, the brain is incredibly complex (the mouse brain has around 100 million neurons [@Herculano-Houzel:2006]) yet has an ordered mesoscopic structure with distinct brain areas delinated by structure, function and connectivity to other brain regions. These brain regions have been studied and collated for many decades, leading to the development of a number of brain atlases. Typically these atlases are made up of a reference image of a brain (or other structure), pixel-wise annotations (e.g. a mapping from each pixel to a brain structure) and additional metadata such as region hierarchy (region A is a subdivision of region B). These atlases are used throughout neuroscience, for teaching, visualisation of results, and registration of imaging data to a common coordinate space.

Many excellent atlases exist, such as the Allen Mouse Brain Common Coordinate Framework [@Wang:2020] and the Max Planck Larval Zebrafish Atlas [@Kunst:2019], and many neuroscientists have built software tools around these. The problem is that not all available atlases have an API, and those that do are not consistent. As the majority of neuroscientists only work with a single model organism, most software tools are developed with only one atlas in mind. 

The BG-atlasAPI was developed to overcome these problems, and provide an interface for developers to access data from multiple atlases in common formats. Each atlas can be instantiated by passing the atlas name to the `BrainGlobeAtlas` class. A number of files are provided as class attributes including a reference (brian structural) image, an annotation image (a map of brain regions coded by voxel intensity), meshes for each brain region, and various metadata such as the authors of the atlas, and the hierarchy of the brain regions.


# Acknowledgements

# References