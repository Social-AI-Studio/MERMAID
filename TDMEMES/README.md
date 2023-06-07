# SG -> TD MEMES.

Note 1 - the deduplication was not successful. There are a lot of duplicates. (comparatively) hence, in annotation, I also labelled the duplicates.
Note 2 - Only SGMEME section was annotated.
Note 3 - On quick inspection, several NON-SGMEME memes are SG memes, and several are also duplicates of each other and in the SGMEME set.

## Steps taken
1. Begin by running "create_TD_annotation.py". Note you can toggle to use nonSG memes in that dataset instead. - This creates "TD_labelstudio_import_SGONLY.json". The name can be toggled within the file itself
2. Take the created import file for label studio, import. Use the interface code located in the interface storage text - TDMEMES_Label_Studio_Interface_Storage.txt
3. Annotate. (OCR Corrections and text grouping/Boundingbox)
4. Export the File. In this case we exported it as "SGMEMES_Corrected_OCR_labelout.json". 
5. We then run - "parse_TDMEMES_OCR.py", which generates "TDMEMES_RelationAnnotationImport_dict.json", for the second set of annotations (relations + Entity)
6. Take the created import file for label studio, import. Use the interface code located in the interface storage text - TDMEMES_Label_Studio_Interface_Storage.txt (second one inside)
7. Annotate. (Entity/Relations)
8. Export the file. In this case we exported it as "SGMEMES_labelout_Annotated_Relation_completed.json".
9. Run "parse_TDMEMES_FULL_ANNOTATED.py". Change the target file to point to "SGMEMES_labelout_Annotated_Relation_completed.json" or what you named the export as. This creates "SGMEMES_dataset_processed_final.json" and "SGMEMES_PROCESSED_COMPLETED_LABELS.json".
10. "SGMEMES_dataset_processed_final.json" is used by the dataset class to load the dataset.
11. Run "tune_on_SGMEMES.py" to tune your already trained model on the SG MEMES dataset. An additional file "SG_splitlists.json" will be generated, saving the splits that were used in training.
12. Files and the such are automatically saved.


## Streamlit Viewer
1. A very minimal streamlit viewer has been created to allow viewing of predictions. This exists for the position abalated version of predictions. 
    - The file name is: "app_streamlit_position_abalated_TD_relations.py" for viewing relations, and "app_streamlit.py" to view the entities.
    ```python -m streamlit run app_streamlit.py```
    ```python -m streamlit run app_streamlit_position_abalated_TD_relations.py```
    