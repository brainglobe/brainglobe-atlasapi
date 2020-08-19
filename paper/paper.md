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
  - name: Luigi Petrucco*
    affiliation: 2
  - name: Adam Tyson*
    orcid: 0000-0003-3225-1130
    affiliation: 1
  - name: Tiago Branco
    affiliation: 1
  - name: Troy Margrie
    affiliation: 1
  - name: Ruben Portugues
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
Neuroanatomical data analysis relies on precise understanding of where in the brain the information comes from. For this reason, brain atlases have been developed which provide brain region annotation overlaid upon a standardized reference brain image. These atlases have been developed for many animal model species, and are routinely used for data analysis and visualisation. The availability of these atlases vary, many are available online and some have an application programming interface (API), but these are inconsistent. This hinders the development and adoption of open-source neuroanatomy software, as each tool is typically only developed for a single atlas for one model organism. The brainglobe atlas API (BG-Atlas API) overcomes this problem by providing a common interface for programmers to download and process data from multiple atlases. Software can then be developed agnostic to the atlas, increasing adoption and interopability of software in neuroscience. 

# Statement of need 
Neuroscience is gradually becoming a field reliant on large scale data acquisition, such as whole-brain optical neural recordings [@Ahrens:2012], brain-wide connectivity mapping [@Osten:2013] or high-density electrophysiological probes [@Jun:2017]. For this reason, custom, computational analysis is now necessary in the majority of neuroscience laboratories, and Python is emerging as the language of choice [@Muller:2015].

In mammals and other vertebrates, the brain is incredibly complex (the mouse brain has around 100 million neurons [@Herculano-Houzel:2006]) yet has an ordered mesoscopic structure with distinct brain areas delineated by structure, function and connectivity to other brain regions. These brain regions have been studied for many decades, leading to the development of a number of brain atlases. Typically these atlases are made up of a reference image of a brain, pixel-wise annotations (e.g. a mapping from each voxel to a brain structure) and additional metadata such as region hierarchy (region A is a subdivision of region B). These atlases are used throughout neuroscience, for teaching, visualisation of data, and registration of imaging data to a common coordinate space.

Many excellent atlases exist, such as the Allen Mouse Brain Common Coordinate Framework [@Wang:2020] and the Max Planck Larval Zebrafish Atlas [@Kunst:2019], and neuroscientists have built software tools around these. The problem is that not all available atlases have an API, and those that do are not consistent. As the majority of neuroscientists only work with a single model organism, most software tools are developed with only one atlas in mind. 

`BG-atlasAPI` was developed to overcome these problems, and provide an interface for developers to access data from multiple atlases in common formats. Each atlas can be instantiated by passing the atlas name to the `BrainGlobeAtlas` class. A number of files are provided as class attributes including a reference (structural) image, an annotation image (a map of brain regions coded by voxel intensity), meshes for each brain region, and various metadata such as the authors of the atlas, and the hierarchy of the brain regions. There are methods for many common tasks such as orienting data and parsing the region hierarchy.

`BG-atlasAPI` was built with two purposes. The first was to simplify the use of atlases for neuroscientists by providing a simple, concise and well-documented API. The second was to reduce the burden required to develop tools that can be used across atlases. The majority of neuroanatomical software tools developed currently are for a single model organism, yet many of these tools could be of great use for many other neuroscientists. 

Currently six atlases across three species (larval zebrafish, mouse and human) are available, with work underway to add further atlases (e.g. rat). We provide scripts for automated generation of these atlases, and welcome contributions from anyone who wants to add an atlas. 

We have used `BG-atlasAPI` to develop two software tools for use in neuroscience, brainreg [@Tyson:2020] for 3D registration of image data between sample and atlas coordinate space and brainrender [@Claudi:2020] for visualisation of user-generated data, and atlas data in a common coordinate space. We hope that other developers will use the API, and develop tools that can be used across neuroscience, increasing their reach, and prevent duplication of effort.

# References