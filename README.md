# MERMAID: A Dataset and Framework for Multimodal Meme Semantic Understanding


If you are just after the dataset [this is the link if you don't want to pull the entire repository](??????????)

## Getting Started

Requirements:
1. 7zip to uncompress all files. (They are compressed using LZMA for space requirements on github.)
2. Download all additional large files hosted at ????. Place them in the correct locations within the repository, or edit file addresses within the code to do so.
3. If you want to emulate/redo annotation, use "requirements_for_labelstudio.txt" to set up a virtual environment. Note that we perform additional corrections for spelling mistakes. The official dataset is above. The code exists to show how we performed annotations, although it has been simplified to only accept a SINGLE annotation file. (Combining of all annotator results was done separately) <br> * don't forget to export and setup the variable that allows for loading images locally. <br>```- set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true (export for mac/linuxlikes)```
4. Download environment at requirements.txt.

* Initial data collection + Annotation + Data management - Go into "parse" folder
* TD MEMES Alternative Exploration - Go into "TDMEMES" Folder.
* To run the model as is: "run_model.py" if all other requirements have been met.
