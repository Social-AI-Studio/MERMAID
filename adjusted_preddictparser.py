import json
import os
import torch
import pprint
import csv
from numpy import argsort
from dataset_class import dual_dataset, entity_error_analysis, dataset_list_split

remapped_actionables_dict = {
"Superior":0,
"Equal":1,
"Upgrade":2,
"Degrade":3,
"Affirm/Favor":4,
"Doubt/Disfavor":5,
"Indifferent":6,
"Inferior":7,
"NULL":8,
}
if __name__=="__main__":
    relationthreshold = 0.5
    entitythreshold = 0.5
    print_incorrect_entities = False
    perform_smartstrike = False
    pause_between_files = False
    entity_model = "bert"
    # entity_model = "roberta"
    targetdir = "results_prata"
    flipped_actiondict = {i:k for k,i in remapped_actionables_dict.items()}
    computed_results_dict = {
        "Learning Rate":[],
        "Batch Size":[],
        "Model Type":[],
        "Entity F1":[],
        "Entity Precision":[],
        "Entity Recall":[],
        "Total Entities":[],
        "Correct Entities":[],
        "Entity Correct Percentage":[],
        "Total Relation Pairs":[],
        "Relation Macro F1":[],
        "Relation Micro F1":[],
        "Relation Micro Precision":[],
        "Relation Micro Recall":[],
        "Relation Micro TP":[],
        "Relation Micro FP":[],
        "Relation Micro FN":[],
        "Relation Micro TN":[],
    }
    
    for targetfile in os.listdir(targetdir):
        if not ".json" in targetfile :
            continue

        print(targetfile)
        
        lr = targetfile.split("_")[2]
        batchsize = targetfile.split("_")[3]
        model_type = targetfile.split("test")[1].replace(".json","")
        
        computed_results_dict["Learning Rate"].append(lr)
        computed_results_dict["Batch Size"].append(batchsize)
        computed_results_dict["Model Type"].append(model_type)
        
        
        mean_average_precision_dict = {}
        
        if perform_smartstrike:
            bestguesser = dataset_list_split("final_dataset.json",0.6,[],return_smartstrike=True)
        
        with open(os.path.join(targetdir,targetfile),"r",encoding="utf-8") as openedfile:
            loaded_results = json.load(openedfile)
        
        allowed_images = []
        for sourceimages in  loaded_results:
            allowed_images.append(sourceimages)
        
        
        dataset_main = dual_dataset("final_dataset.json",target_tokeniser=["bert","roberta"],verbose=False,approved_images=allowed_images)
        
        totalentities = 0
        pureentities = 0
        correctentities = 0 
        total_relation_pairs = 0
        relation_classwisedict = {}
        relation_classwisef1dict = {}
        for idx in range(9):
            relation_classwisef1dict[idx] = {"FP":0,"TP":0,"TN":0,"FN":0}
            relation_classwisedict[idx] = {"correct":0,"wrong":0,"total":0}
        archetype_recorder = {}
        entity_f1_dict = {"FP":0,"TP":0,"TN":0,"FN":0}
        for data_out in dataset_main:
            if not data_out["source_image"] in loaded_results:
                continue
            imagefilename = data_out["source_image"]
            
            
            
            archetype = loaded_results[imagefilename]["archetype"]
            
            
            if not archetype in mean_average_precision_dict:
                mean_average_precision_dict[archetype] = {"total":0,"average_precision_list":[]}
            
            if not archetype in archetype_recorder:
                archetype_recorder[archetype] = {"total":0,"correct":0,"internaldict":{}}
            
            relations = loaded_results[imagefilename]["relations"]
            for relationpair in relations:
                if relations[relationpair]["idSENDER"]=="MEME_CREATOR":
                    relations[relationpair]["idSENDER"] = data_out["meme_creator"]
                if relations[relationpair]["idRECEIVER"]=="MEME_CREATOR":
                    relations[relationpair]["idRECEIVER"] = data_out["meme_creator"]
                correct_targets = data_out["inverted_numerical_relationships"][(relations[relationpair]["idSENDER"],relations[relationpair]["idRECEIVER"])]
                selected_targets = []
                
                argsort_logits = list(reversed(argsort(relations[relationpair]["Prediction (Logits)"])))
                
                
                
                seencount = 0
                num_correct_targets = len(correct_targets)
                summer = 0
                for argsortidx in range(len(argsort_logits)):
                    if argsort_logits[argsortidx] in correct_targets:
                        seencount+=1
                        summer+=seencount/(argsortidx+1)
                mean_average_precision_dict[archetype]["average_precision_list"].append(summer)
                mean_average_precision_dict[archetype]["total"] = mean_average_precision_dict[archetype]["total"] + 1
                
                
                
                if perform_smartstrike:
                    selected_targets= list(bestguesser[data_out["archetype"]])
                else:
                    for idx in range(len(relations[relationpair]["Prediction (Logits)"])):
                        if relations[relationpair]["Prediction (Logits)"][idx]>relationthreshold:
                            selected_targets.append(idx)
                    
                for idx in range(9):
                    if not idx in archetype_recorder[archetype]["internaldict"]:
                        archetype_recorder[archetype]["internaldict"][idx] = {"FP":0,"TP":0,"TN":0,"FN":0}
                
                archetype_recorder[archetype]["total"] = archetype_recorder[archetype]["total"] + 1
                total_relation_pairs+=1
                
                
                for selected in selected_targets:

                    if selected in correct_targets: # True positive
                        relation_classwisef1dict[selected]["TP"] = relation_classwisef1dict[selected]["TP"] + 1
                        archetype_recorder[archetype]["internaldict"][selected]["TP"] = archetype_recorder[archetype]["internaldict"][selected]["TP"] + 1
                        
                        
                    elif not selected in correct_targets: # false positive
                        relation_classwisef1dict[selected]["FP"] = relation_classwisef1dict[selected]["FP"] + 1
                        archetype_recorder[archetype]["internaldict"][selected]["FP"] = archetype_recorder[archetype]["internaldict"][selected]["FP"] + 1
                        # print(imagefilename, "FP",archetype)
                        # print(data_out["text_locs"])
                        # input()
                        
                for target in correct_targets:
                    if not target in selected_targets: # False Negative
                        relation_classwisef1dict[target]["FN"] = relation_classwisef1dict[target]["FN"] + 1
                        archetype_recorder[archetype]["internaldict"][target]["FN"] = archetype_recorder[archetype]["internaldict"][target]["FN"] + 1
                        # print(imagefilename, "FN",archetype)
                        # print(data_out["text_locs"])
                        # input()
                        
                for idx in range(9):
                        
                    if not idx in correct_targets and not idx in selected_targets:
                        relation_classwisef1dict[idx]["TN"] = relation_classwisef1dict[idx]["TN"] + 1 
                        archetype_recorder[archetype]["internaldict"][idx]["TN"] = archetype_recorder[archetype]["internaldict"][idx]["TN"] + 1
                
                
                for correct in correct_targets:
                    if correct in selected_targets:
                        relation_classwisedict[correct]["correct"] = relation_classwisedict[correct]["correct"] + 1
                        archetype_recorder[archetype]["correct"] = archetype_recorder[archetype]["correct"] + 1
                    else:
                        relation_classwisedict[correct]["wrong"] = relation_classwisedict[correct]["wrong"] + 1
                    relation_classwisedict[correct]["total"] = relation_classwisedict[correct]["total"] + 1

            entities = loaded_results[imagefilename]["entities"]
            
            incorrect_entity_flag = False
            
            for textbox in entities:
                correct_entity_ans = data_out["correct_answers"][textbox][entity_model]
                predactual = torch.tensor(entities[textbox]["predicted_actual"])
                span_answer = entities[textbox]["actual"]
                # print(entities[textbox]["text"])
                report,_ = entity_error_analysis(predactual,correct_entity_ans,span_answer,entitythreshold,device="cpu",loss_entity=False,transcendence=False)
                
                for entity_tensor_idx in range(len(predactual)):
                    if correct_entity_ans[entity_tensor_idx][0]==1:
                        if predactual[entity_tensor_idx][0]>entitythreshold:
                            entity_f1_dict["TP"] +=1
                        else:
                            entity_f1_dict["FN"] +=1
                    else:
                        if predactual[entity_tensor_idx][0]>entitythreshold:
                            entity_f1_dict["FP"] +=1
                        else:
                            entity_f1_dict["TN"] +=1
                        
                
                # pprint.pprint(report)
                for entity in report:
                    if report[entity]["start"] and report[entity]["stop"] and not report[entity]["midfailure"]:
                        pureentities+=1
                    else:
                        incorrect_entity_flag =True
                        # print("Wrong Entity:",imagefilename)
                        # input()
                    # if report[entity]["start"] and report[entity]["stop"]:
                        # correctentities+=1
                        
                    totalentities+=1
                
            if incorrect_entity_flag and print_incorrect_entities:
                print(archetype)
                print("Incorrect Entity in:",imagefilename, report)
                print()
                
                
                
        print(entity_f1_dict)
        if entity_f1_dict["TP"] + entity_f1_dict["FP"]==0:
            entity_precision = 0
        else:
            entity_precision = entity_f1_dict["TP"]/ (entity_f1_dict["TP"] + entity_f1_dict["FP"])
        if entity_f1_dict["TP"] + entity_f1_dict["FN"]==0:
            entity_recall = 0
        else:
            entity_recall = entity_f1_dict["TP"]/ (entity_f1_dict["TP"] + entity_f1_dict["FN"])
        
        if entity_precision+entity_recall==0:
            entity_f1_score = 0
        else:
            entity_f1_score = 2*entity_precision*entity_recall /(entity_precision + entity_recall)
        print("Entity F1 Score",entity_f1_score)
        print("Entity Precision",entity_precision)
        print("Entity Recall",entity_recall)
        computed_results_dict["Entity F1"].append(entity_f1_score)
        computed_results_dict["Entity Precision"].append(entity_precision)
        computed_results_dict["Entity Recall"].append(entity_recall)
        
        

    
        
        
        print("Total entities:",totalentities)
        print("Correct entities:",pureentities)
        computed_results_dict["Total Entities"].append(totalentities)
        computed_results_dict["Correct Entities"].append(pureentities)
        
        # print("Correct(onlyflag) entities:",correctentities)
        print("Correct entities Percentage:",round(pureentities/totalentities,4)*100,"%")
        # print("Correct(onlyflag) entities Percentage:",round(correctentities/totalentities,4)*100,"%")    
        print("Total Relation Pairs:",total_relation_pairs)
        computed_results_dict["Entity Correct Percentage"].append(round(pureentities/totalentities,4)*100)
        computed_results_dict["Total Relation Pairs"].append(total_relation_pairs)
        
        
        flipped_actiondict = {i:k for k,i in remapped_actionables_dict.items()}
        print("-"*30)
        macrof1summer = 0
        counter = 0
        
        
        for relation in relation_classwisedict:
            # print(flipped_actiondict[relation])
            # print("Absolute correctness dict")
            # pprint.pprint(relation_classwisedict[relation])
            print("Correct:",relation_classwisedict[relation]["correct"])
            print("Wrong:",relation_classwisedict[relation]["wrong"])
            # print("Relation classwise F1 dict:")
            
            # pprint.pprint(relation_classwisef1dict[relation])
            if relation_classwisef1dict[relation]["TP"] + relation_classwisef1dict[relation]["FP"]==0:
                precision = 0
            else:
                precision = relation_classwisef1dict[relation]["TP"]/ (relation_classwisef1dict[relation]["TP"] + relation_classwisef1dict[relation]["FP"])
            if relation_classwisef1dict[relation]["TP"] + relation_classwisef1dict[relation]["FN"]==0:
                recall = 0
            else:
                recall = relation_classwisef1dict[relation]["TP"]/ (relation_classwisef1dict[relation]["TP"] + relation_classwisef1dict[relation]["FN"])
            
            if precision+recall==0:
                f1_score = 0
            else:
                f1_score = 2*precision*recall /(precision + recall)
            print(flipped_actiondict[relation],relation_classwisef1dict[relation]["TP"],relation_classwisef1dict[relation]["TN"],relation_classwisef1dict[relation]["FP"],relation_classwisef1dict[relation]["FN"],f1_score)
            # print("F1 score:",f1_score)
            macrof1summer +=f1_score
            counter+=1
            print("-"*30)
            
            if not flipped_actiondict[relation]+" F1" in computed_results_dict:
                computed_results_dict[flipped_actiondict[relation]+" F1"] = []
                computed_results_dict[flipped_actiondict[relation]+" Recall"] = []
                computed_results_dict[flipped_actiondict[relation]+" Precision"] = []
            computed_results_dict[flipped_actiondict[relation]+" F1"].append(f1_score)
            computed_results_dict[flipped_actiondict[relation]+" Recall"].append(recall)
            computed_results_dict[flipped_actiondict[relation]+" Precision"].append(precision)
            
            
        print("Macro F1:",macrof1summer/counter)
        computed_results_dict["Relation Macro F1"].append(macrof1summer/counter)
        
        massTP = 0
        massFP = 0
        massFN = 0
        massTN = 0
        
        for arch in archetype_recorder:
            
            totalTP = 0
            totalFP = 0
            totalFN = 0
            totalTN = 0
            for item in list(archetype_recorder[arch]["internaldict"].keys()):
                totalTP += archetype_recorder[arch]["internaldict"][item]["TP"]
                totalFP += archetype_recorder[arch]["internaldict"][item]["FP"]
                totalFN += archetype_recorder[arch]["internaldict"][item]["FN"]
                totalTN += archetype_recorder[arch]["internaldict"][item]["TN"]
                
                
                if archetype_recorder[arch]["internaldict"][item]["TP"] + archetype_recorder[arch]["internaldict"][item]["FP"] ==0:
                    precision = 0
                else:
                    precision = archetype_recorder[arch]["internaldict"][item]["TP"]/ (archetype_recorder[arch]["internaldict"][item]["TP"]+archetype_recorder[arch]["internaldict"][item]["FP"])
                
                if archetype_recorder[arch]["internaldict"][item]["TP"] + archetype_recorder[arch]["internaldict"][item]["FN"] ==0:
                    recall = 0 
                else:
                    recall = archetype_recorder[arch]["internaldict"][item]["TP"]/(archetype_recorder[arch]["internaldict"][item]["TP"] + archetype_recorder[arch]["internaldict"][item]["FN"])
                
                if precision+recall==0:
                    archetype_recorder[arch]["internaldict"][item]["f1_score"] = 0
                else:
                    archetype_recorder[arch]["internaldict"][item]["f1_score"] = 2*precision*recall /(precision + recall)
            
            
            archetype_recorder[arch]["TP"] = totalTP
            archetype_recorder[arch]["TN"] = totalTN
            archetype_recorder[arch]["FP"] = totalFP
            archetype_recorder[arch]["FN"] = totalFN
            massTP +=totalTP
            massFP +=totalFP
            massFN += totalFN
            massTN += totalTN
            
            if totalTP + totalFP==0:
                precision = 0
            else:
                precision = totalTP/(totalTP + totalFP)

            if totalTP + totalFN==0:
                recall=0
            else:
                recall = totalTP/(totalTP + totalFN)
            
            if precision+recall==0:
                f1score = 0
            else:
                f1score = 2*precision*recall /(precision + recall)
            archetype_recorder[arch]["f1score"] = round(f1score,4)
            
            if not arch+" F1" in computed_results_dict:
                computed_results_dict[arch+" F1"] = []
                computed_results_dict[arch+" Recall"] = []
                computed_results_dict[arch+" Precision"] = []
                computed_results_dict[arch+" TP"] = []
                computed_results_dict[arch+" FP"] = []
                computed_results_dict[arch+" FN"] = []
                computed_results_dict[arch+" TN"] = []
            computed_results_dict[arch+" F1"].append(f1score)
            computed_results_dict[arch+" Recall"].append(recall)
            computed_results_dict[arch+" Precision"].append(precision)
            computed_results_dict[arch+" TP"].append(totalTP)
            computed_results_dict[arch+" FP"].append(totalFP)
            computed_results_dict[arch+" FN"].append(totalFN)
            computed_results_dict[arch+" TN"].append(totalTN)
            
            
        
        overall_count = 0
        overall_precision = 0
        
        for arch in sorted(list(mean_average_precision_dict.keys())):
            archresult = sum(mean_average_precision_dict[arch]["average_precision_list"])/mean_average_precision_dict[arch]["total"]
            overall_precision += sum(mean_average_precision_dict[arch]["average_precision_list"])
            overall_count += mean_average_precision_dict[arch]["total"]
            print(arch,"Mean Average Precision",archresult)
        
        print("Total Mean Average Precision:",overall_precision/overall_count)
        
        print("-"*50)
        print("\n\n")
        
        masscorrect = 0
        masstotal = 0
        for arch in sorted(list(archetype_recorder.keys())):
            print(arch,"overall_relationship accuracy:",round(archetype_recorder[arch]["correct"]/archetype_recorder[arch]["total"]*100,2))
            masstotal+= archetype_recorder[arch]["total"]
            masscorrect += archetype_recorder[arch]["correct"]
        print("Overall mass accuracy",masscorrect/masstotal*100)
            
        print("\n\n\n\n")
        for arch in sorted(list(archetype_recorder.keys())):
            print(arch,"f1 score: ",archetype_recorder[arch]["f1score"])
        
        print(massTP,massFP,massFN,massTN)
        
        if massTP + massFP==0:
            precision = 0
        else:
            precision = massTP/(massTP + massFP)

        if massTP + massFN==0:
            recall=0
        else:
            recall = massTP/(massTP + massFN)
        
        if precision+recall==0:
            f1score = 0
        else:
            f1score = 2*precision*recall /(precision + recall)
        print("Micro Precision:",precision)
        print("Micro Recall:",recall)
        print("Micro F1:",f1score)
        computed_results_dict["Relation Micro F1"].append(f1score)
        computed_results_dict["Relation Micro Precision"].append(precision)
        computed_results_dict["Relation Micro Recall"].append(recall)
        computed_results_dict["Relation Micro TP"].append(massTP)
        computed_results_dict["Relation Micro FP"].append(massFP)
        computed_results_dict["Relation Micro TN"].append(massTN)
        computed_results_dict["Relation Micro FN"].append(massFN)
        
        if pause_between_files:
            input()
        
        
    pprint.pprint(computed_results_dict)
    with open("Comparison_file.csv","w",encoding="utf-8",newline="") as comparisondumpfile:
        writer = csv.DictWriter(comparisondumpfile,fieldnames=list(computed_results_dict.keys()))
        writer.writeheader()
        for computed_targetkey in range(len(computed_results_dict[list(computed_results_dict.keys())[0]])):
            dumpdict = {}
            for results_dict_key in computed_results_dict:
                dumpdict[results_dict_key] = computed_results_dict[results_dict_key][computed_targetkey]
            writer.writerow(dumpdict)
    # pprint.pprint(archetype_recorder)
        
        
        
        