# IHC_Synovium_Segmentation

--------------

Gallagher-Syed A., Khan A., Rivellese F, Pitzalis C., Lewis M. J., Slabaugh G., Barnes M., "Automated segmentation of rheumatoid arthritis immunohistochemistry stained synovial tissueL", _Medical Image Understanding and Analysis 2023_, Aberdeen. 2023. <a href="https://github.com/AmayaGS/HistoMIL/blob/ac2ae80b998afc4f7298161562dba8bf2f688a4a/Automated_segmentation_of_Rheumatoid_Arthritis_Immunohistochemistry_stained_synovial_tissue.pdf" target="_blank">Conference abstract.</a>

--------------

A modified CLAM/MIL pipeline for large histopathology WSIs, composed of:

<ol>
  <li>A VGG16 embedding backbone, previously trained on the WSI patches with labels propagated from the slide level, reducing each patch to a 1024 feature vector. </li>
  <li>All embeddings from a slide are aggregated into a larger feature vector and passed to the attention model</li>
  <li>The attention modell (CLAM/MIL) aggregates the patch-level information to the slide level, as well as calculating an attention score for each patch</li>
  <li>A slide level prediction is made</li>
  <li>A heatmap is created for the slide</li>
</ol>

![alt text](https://github.com/AmayaGS/HistoMIL/blob/main/model.png?raw=true)

