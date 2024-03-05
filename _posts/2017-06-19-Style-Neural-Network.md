---
layout: page
title:  "Application Development"
date:   2023-08-15 10:17:15 -0500
---

### Interactive Visualization of Mangrove Cover Changes using eCognition's User Interactive Architecture

This project focuses on the analysis of mangrove coverage in Bintang Bolong, a beautiful and ecologically significant area located in The Gambia. Mangroves play a vital role in coastal ecosystems, providing various ecological and socioeconomic benefits. The Mangrove Vegetation Index (MVI) is an important tool for quantifying and monitoring mangrove coverage, and this project demonstrates the calculation and analysis of MVI using satellite imagery. Bintang Bolong is a stunning mangrove-rich estuary located in the southern part of The Gambia. It is known for its diverse mangrove species and serves as a crucial habitat for a wide range of flora and fauna. The region is not only ecologically significant but also of great importance to the local communities that rely on mangroves for sustenance and livelihoods.

### Research
This work is an extension of my previous work (Source: Ligono, L. K. and Okolie, C. J.: INTEGRATED ANALYSIS OF MANGROVE CHANGES USING THE MANGROVE VEGETATION INDEX AND RANDOM FOREST CLASSIFICATION IN THE GAMBIA, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLVI-M-2-2022, 153–157, https://doi.org/10.5194/isprs-archives-XLVI-M-2-2022-153-2022, 2022.) that I conducted in mapping mangroves using pixel-based classification.

### Workflow

Step 1: a) Load Sentinel-2 image of the Bintang Bolong Estuary in eCognition b) Create a ruleset to automatically rename the bands of the Sentinel-2 image to Blue, Red, Green, NIR, SWIR.

Step 2: Develop a ruleset in eCognition for conducting Random Forest classification and the the Mangrove Vegetation Index (MVI) using the relevant spectral bands.

Step 3: Design an interactive architecture in eCognition that enable users to visualize either the MVI or the results of Random Forest classification. 

### Results
![cover_page](https://github.com/lisahligono/OBIA_IP/assets/72496335/5015a7af-5926-4622-955d-44cbec5c0538){:width="700px"}

### Getting started with the User Architecture 

To use this application, you require a licenced eCognition version 10.1 and above software.
- Open the eCognition Developer interface
- Click on *Architect* --> select '**Open Action Library**' and select the folder with the action library. In this case the folder is named '**Ligono_architecture**'
- Next click again on *Architect* and select '**Load solution**'--> select '**interfacev3.dax**' and the analysis builder will load as seen below:
  
![builder](https://github.com/lisahligono/OBIA_IP/assets/72496335/5cd29ffa-408f-4580-bc79-7aa81f47cdca){:width="700px"}

- Load the Sentinel-2 image in the folder
- Now you can start using the user architecture: Click on the Multi-resolution Segmentation tab and set the scale parameter  by moving the slider. (The example below show segmentation with a segmentation of 500)

![segmentation](https://github.com/lisahligono/OBIA_IP/assets/72496335/83ae362e-36c1-4e1e-bcf1-1ec98dc92a53){:width="700px"}
 
- Once the segmentation is complete collect samples for all the classes in the 'Class Hierarchy' tab except 'mvi_mangrove'

 ![samples](https://github.com/lisahligono/OBIA_IP/assets/72496335/5d2962c8-a34e-4ab7-9e23-ccc359b57440){:width="700px"}
 
- Click on the Random Forest Algorithm tab and execute Random Forest --> the result will look like this

  ![rf](https://github.com/lisahligono/OBIA_IP/assets/72496335/24dedb13-e513-43db-9c66-cc8a58efbe07){:width="700px"}

- If you want to visualise the classification from the MVI, execute Calculate MVI

  ![mvi](https://github.com/lisahligono/OBIA_IP/assets/72496335/6b11f3bf-fed7-48e0-93cf-25e21493b5a3){:width="700px"}


  And there you have it, an interactive user architecture for mangrove mapping 😁










