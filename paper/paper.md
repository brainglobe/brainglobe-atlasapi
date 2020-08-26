---
title: 'Brainglobe atlas API: a common interface for neuroanatomical atlases'
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
  - name: Adam L. Tyson*
    orcid: 0000-0003-3225-1130
    affiliation: 1
  - name: Tiago Branco
    affiliation: 1
  - name: Troy W. Margrie
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
Neuroscientists routinely perform experiments aimed at recording or manipulating neural activity, uncovering physiological proesses underlying brain function or elucidating aspects of brain anatomy. Understanding how the brain generates behaviour ultimately depends on the possibility to merge the results of these experiments into a coherent picture of brain anatomy and function. Brain atlases are crucial in this endeavour: by outlining the organization of brain regions they provide a reference upon which our understanding of brain function can be anchored. More recently, digital high-resolution 3d atlases have been produced for several model organisms providing an invaluable resource for the research community. 
Effective use of these atlases depends on the availability of an application programming interface (API) which can be used to develop software accessing atlas data. However, while some atlases come with an API, these are generally incosistent across atlases.
The lack of an unified atlas API hinders the development and adoption of open-source neuroanatomy software, as each tool is typically only developed for a for a single atlas and a particular model organism and rarely tested across different species. The brainglobe atlas API (BG-Atlas API) overcomes this problem by providing a common interface for programmers to download and process data from multiple atlases. By adopting the BG-Atlas API, software can then be developed agnostic to the atlas, increasing adoption and interoperability of packages in neuroscience and enabling direct integration of different experimental modalities and even comparisons across model organisms. 

# Statement of need 
To facilitate the study of neural function, a long standing approach has been to identify neuroanatomically defined brain regions: structures with defined function, connectivity and anatomical location. The study of these brain regions led to the development of a number of brain atlases for various species. Typically these atlases are made up of a reference image of a brain, voxel-wise annotations (e.g. a mapping from each voxel to a brain structure) and additional metadata such as region hierarchy (region A is a subdivision of region B). These atlases are used throughout neuroscience, for teaching, visualisation of data, and registration of imaging data to a common coordinate space [@Randlett:2015].

Many excellent and open access atlases exist, such as the Allen Mouse Brain Common Coordinate Framework [@Wang:2020] and the Max Planck Larval Zebrafish Atlas [@Kunst:2019], from which the neuroscience community benefits enormously. Many atlases provide an API which is crucial for the development of programming tools that use these atlases, however not all atlases provide an API and there is no common API which can work across atlases, hindering the development of tools for neuroanatomical analysis that can work on different atlases. 
Notably, within the R ecosystem some effort has been made in the direction of a common API for quantitative neuroanatomy analysis with the `natverse` package [@Shakeel-Bates:2020], but, to our knowledge, no such option exists in Python, which is emerging as the programming language of choice in neuroscience [@Muller:2015].

`BG-atlasAPI` was built to address these issues and with two main design goals in mind. The first was to simplify the use of atlases for neuroscientists by providing a simple, concise and well-documented API. The second was to reduce the burden required to develop tools that can be used across atlases. The majority of neuroanatomical software tools developed currently are for a single model organism, yet many of these tools could be of great use for many other neuroscientists. 

Developers can use `BG-atlasAPI` to access data from multiple atlases in common formats. Each atlas can be instantiated by passing the atlas name to the `BrainGlobeAtlas` class. A number of files are provided as class attributes including a reference (structural) image, an annotation image (a map of brain regions coded by voxel intensity), meshes for each brain region, and various metadata such as the authors of the atlas, and the hierarchy of the brain regions. There are methods for many common tasks such as orienting data and parsing the region hierarchy.

Currently six atlases across three species (larval zebrafish, mouse and human) are available [@Wang:2020; @Ding:2016; @Kunst:2019; @Chon:2019], with work underway to add further atlases (e.g. rat, drosophila). We provide scripts for automated generation of these atlases, and welcome contributions from anyone who wants to add an atlas. 

We have used `BG-atlasAPI` to develop two software tools for use in neuroscience, `brainreg` [@Tyson:2020] for 3D registration of image data between sample and atlas coordinate space and `brainrender` [@Claudi:2020] for visualisation of both user-generated data and atlas data. We hope that other developers will use the API, and develop tools that can be used across neuroscience, increasing their reach, and preventing duplication of effort.

# References
