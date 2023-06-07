# Preprocessing of raw data and annotation

That's what this folder is composed of. Allows you to recreate the data from scratch.



1. raw_data_jsons contains the pulling script. You will need selenium to pull. The selenium version should be 4.1.3. Be very careful about selenium versions they did major changes to syntax that don't transfer across major releases.
2. raw_data_jsons should contain "image_dir" and "template" folders. These contain your meme template files and your memes respectively.
3. raw_data_jsons should also contain the rest of the meme information that was collected when it was pulled.
4. Data should be available ???????????
5. mass_tokenizer_class.py stores many utilities for the recreation of the annotations. <br>
    * The annotations occurred via the following steps 
    * Step 1 -> Spelling Correction + Grouping of text  - Generate "label_studio_reference_input_OCR_INITIAL", import into label studio and annotate.
    * Step 2-> Bounding Box Corrections - Generate "label_studio_reference_input_BoundingBox_Corrections.json", import into label studio and annotate.
    * Step 3-> Relations + Entity Annotations - Generate "label_studio_reference_input_Relations_Entities.json", import into label studio and annotate.
    * Step 4 -> Compose into dataset jsons. compose the data into "Processed_annotations.json" and "final_dataset_cleared.json"
6. All annotation interface html is located inside "Label_Studio_Interface_Storage.txt". Use the correct interface when labelling in label studio. (Do not change the field names).
7. After annotation is complete run "run_model.py". You can set which optimizer lrs you want to run, and what batch sizes. (Leave them in their respective lists.)
 * If target_fewshot_templates is not an empty list, "fewshot_targetks" will come into play, and allow you to select which fewshot ks you want to run with. It is ignored otherwise.
 * There are 3 different entity embeddings and 4 different relation embeddings possible. They are set manually in lines 130-147.
 * When all settings are done, you can run the "file run_model"
8. If you want to perform a comparison on raw data as to the effects of OCR corrections on entity accuracy alone, use "OCR_vs_RAW_comparison.py" 
9. If you want code that hard checks the total counts of relations and entities (to check it tallies with the published work), use: "count_dataset_values.py"
10. To perform a run on the dataset with RAW OCR instead of corrected/grouped ocr, it is also possible if you've already trained a model. run "direct_OCR_result_vs_annotated_compile.py" to generate "OCR_valid_images_list.json", a json containing all images that have at least ONE entity correct in the OCR.
    * run "run_OCRBOX_as_textboxes.py" and you will obtain your results. Toggle the paths to the correct model inside.


### Important
For all files, there is a variable: noposition. If True, the expected/trained model is a model with position values abalated.