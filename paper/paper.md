---
title: 'BrainGlobe Atlas API: a common interface for neuroanatomical atlases'
tags:
  - Python
  - neuroscience
  - neuroanatomy
  - microscopy

authors:
  - name: Federico Claudi^[Joint first author, ordered alphabetically]
    affiliation: 1
  - name: Luigi Petrucco*
    affiliation: "2, 3"
  - name: Adam L. Tyson*
    orcid: 0000-0003-3225-1130
    affiliation: 1
  - name: Tiago Branco
    affiliation: 1
  - name: Troy W. Margrie
    affiliation: 1
  - name: Ruben Portugues
    affiliation: "2, 3, 4"

affiliations:
 - name: Sainsbury Wellcome Centre, University College London, London, U.K.
   index: 1
 - name: Institute of Neuroscience, Technical University of Munich, Munich, Germany
   index: 2
 - name:  Max Planck Institute of Neurobiology, Research Group of Sensorimotor Control, Martinsried, Germany
   index: 3
 - name: Munich Cluster for Systems Neurology (SyNergy), Munich, Germany
   index: 4
date: 19 August 2020
bibliography: paper.bib

---

# Summary
Neuroscientists routinely perform experiments aimed at recording or manipulating neural activity, uncovering physiological processes underlying brain function or elucidating aspects of brain anatomy. Understanding how the brain generates behaviour ultimately depends on merging the results of these experiments into a unified picture of brain anatomy and function. Brain atlases are crucial in this endeavour: by outlining the organization of brain regions they provide a reference upon which our understanding of brain function can be anchored. More recently, digital high-resolution 3d atlases have been produced for several model organisms providing an invaluable resource for the research community. 
Effective use of these atlases depends on the availability of an application programming interface (API) that enables researchers to develop software to access and query atlas data. However, while some atlases come with an API, these are generally specific for individual atlases, and this hinders the development and adoption of open-source neuroanatomy software. 
The BrainGlobe atlas API (BG-Atlas API) overcomes this problem by providing a common interface for programmers to download and process data  across a variety of model organisms. By adopting the BG-Atlas API, software can then be developed agnostic to the atlas, increasing adoption and interoperability of packages in neuroscience and enabling direct integration of different experimental modalities and even comparisons across model organisms. 

# Statement of need 
To facilitate the study of neural function, a long-standing approach has been to identify neuroanatomically defined brain regions: structures with defined function, connectivity and anatomical location. The study of these brain regions led to the development of a number of brain atlases for various species. Typically these atlases are made up of a reference image of a brain, voxel-wise annotations (e.g. a mapping from each voxel to a brain structure) and additional metadata such as region hierarchy (region A is a subdivision of region B). These atlases are used throughout neuroscience, for teaching, visualisation of data, and registration of imaging data to a common coordinate space.

Many excellent and open access atlases exist, such as the Allen Mouse Brain Common Coordinate Framework [@Wang:2020] and the Max Planck Larval Zebrafish Atlas [@Kunst:2019], from which the neuroscience community benefits enormously. These atlases provide a valuable resource for individual scientist and enabled important open-science projects such as Janelia Campus' Mouse Light project [@Winnubst:2019].  Furthermore, for several atlases stand-alone software is available that can be used to explore the atlas' data and requires no coding experience, thus making the atlases accessible to a broader audience. 
However, to be used in the context of new software (e.g. new visualization tools, or brain registration pipelines)  it is necessary that atlases expose their data through an API. 
Several commonly used atlases come with APIs, but learning how to use each of them is a time-consuming endavour and can require considerable coding experience. For this reason, often developers produce software that works only with a specific atlas. 
A single and well documented API that worked across atlases would thus lower the cost of developing new software, which can also be made available for a larger number of scientists. An effort in this direction has been made in the R ecosystem  with the `natverse` package [@Shakeel-Bates:2020], but, to our knowledge, no such option exists in Python, which is emerging as the programming language of choice in neuroscience [@Muller:2015].

`bg-atlasapi` was built to address these issues and with two main design goals in mind. The first was to simplify the use of atlases for neuroscientists by providing a simple, concise and well-documented API. The second was to reduce the burden required to develop tools that can be used across atlases. The majority of neuroanatomical software tools developed currently are for a single model organism, yet many of these tools could be of great use for many other neuroscientists. 

Developers can use `bg-atlasapi` to access data from multiple atlases in common formats. Each atlas can be instantiated by passing the atlas name to the `BrainGlobeAtlas` class. A number of files are provided as class attributes including a reference (structural) image, an annotation image (a map of brain regions coded by voxel intensity), meshes for each brain region, and various metadata such as the authors of the atlas, and the hierarchy of the brain regions. There are methods for many common tasks such as orienting data and parsing the region hierarchy.

Currently six atlases across three species (larval zebrafish, mouse and human) are available [@Wang:2020; @Ding:2016; @Kunst:2019; @Chon:2019], with work underway to add further atlases (e.g. rat, drosophila). The available atlases were created by parsing their relative online sources and restructuring the data to a standard format. The atlases were then made accessible by hosting the data in a GNode respository (`https://gin.g-node.org/brainglobe/atlases`). The python code used for generating the atlases is also made available in a separate repository in the BrainGlobe organization: `bg-atlasgen`. The same code can be used for easily developing new atlases in BG-AtlasAPI's format and we encourage users to contribute new atlases to the project by submitting new scripts to `bg-atlasgen`.

`BG-atlasAPI`'s flexible infrastructure already proved crucial in the development and extension of two software tools for use in neuroscience: `brainreg` [@Tyson:2020] for 3D registration of image data between sample and atlas coordinate space and `brainrender` [@Claudi:2020] for 3D visualisation of both user-generated data and atlas data.
We hope that other developers will use the API, and develop tools that can be used across neuroscience and other research fields, increasing their reach, and preventing duplication of effort.

# Acknowledgments
We would like to thank Nouwar Mokayes for the assistance in packaging the Max Planck Zebrafish Brain Atlas within `BG-atlasAPI`. This work was supported by grants from the Gatsby Charitable Foundation (GAT3361, T.W.M. and T.B.), Wellcome Trust (090843/F/09/Z, T.W.M. and T.B.; 214333/Z/18/Z, T.W.M.; 214352/Z/18/Z, T.B.) and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy within the framework of the Munich Cluster for Systems Neurology (EXC 2145 SyNergy – ID 390857198).

# References
