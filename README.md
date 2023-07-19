# Immunohistochemistry synovial tissue UNet segmentation

--------------

Gallagher-Syed A., Khan A., Rivellese F, Pitzalis C., Lewis M. J., Slabaugh G., Barnes M., "Automated segmentation of rheumatoid arthritis immunohistochemistry stained synovial tissueL", _Medical Image Understanding and Analysis 2023_, Aberdeen. 2023. <a href="https://github.com/AmayaGS/IHC_Synovium_Segmentation/blob/ac2ae80b998afc4f7298161562dba8bf2f688a4a/Automated_segmentation_of_Rheumatoid_Arthritis_Immunohistochemistry_stained_synovial_tissue.pdf" target="_blank">Conference abstract paper.</a>

--------------

We provide fully trained UNet segmentation weights for WSI IHC synovial tissue which can be used as the first step in an automated image analysis pipeline. It is robust to common WSIs artefacts, clinical centre/scanner batch effect and can be used on different types of IHC stains. It can be used as is, or fine-tuned on any IHC musculoskeletal dataset, removing the need for manual segmentation by pathologists and offering a solution to the current image analysis bottleneck. 

### I am currently cleaning some up some of the code and updating the latest weights, to upload them ASAP. Watch this space! ##

#### Data collection ### 

A total of 164 patients, fulfilling the 2010 American College of Rheumatology/European Alliance of Associations for Rheumatology (EULAR) classification criteria for RA were recruited to the _R4RA clinical trial_ from 20 European centers [15] [7]. Patients underwent ultrasound-guided synovial biopsy of a clinically active joint. Samples were then fixed in formalin, embedded in paraffin, cut with microtome and stained with the relevant IHC stains: IHC CD20 (B cells), IHC CD68 (macrophages) and IHC CD138 (plasma cells) [7]. Samples were then placed on glass slides and scanned into Whole Slide Image (.ndpi format) with digital scanners under 40x or 20x objectives. Below we show representative examples of the three IHC stains used to train the UNet algorithm:

![alt text](https://github.com/AmayaGS/IHC_Synovium_Segmentation/blob/main/histo_pathotype.PNG?raw=false)

Here we show some results:

![alt text](https://github.com/AmayaGS/IHC_Synovium_Segmentation/blob/main/Figure2.png?raw=false)


