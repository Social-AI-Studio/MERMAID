import os
import csv
import json
import torch
import pprint
from numpy import argsort
from tune_on_SGMEMES import dual_dataset
from dataset_class import entity_error_analysis, dataset_list_split

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
    
    entity_model = "bert"
    flipped_actiondict = {i:k for k,i in remapped_actionables_dict.items()}

    computed_results_dict = {
        "Learning Rate":[],
        "Batch Size":[],
        "Model Type":[],
        "Type":[],
        "split":[],
        "Total Entities":[],
        "Correct Entities":[],
        "Entity Correct Percentage":[],
        "Total Relation Pairs":[],
        "Relation Accuracy":[],
        "Relations Correct":[],
        "Relation Macro F1":[],
        "Relation Micro F1":[],
        "Relation Micro Precision":[],
        "Relation Micro Recall":[],
        "Relation Micro TP":[],
        "Relation Micro FP":[],
        "Relation Micro FN":[],
        "Relation Micro TN":[],
    }
    

    idx = 29
    possiblelr = [1,2,5]
    possiblebatch = [4,8,16,32]
    models = ["abalated","normal"]
    splitnum = [0,1,2]
    trainortest = ["train","test"]
    for lr in possiblelr:
        for batchsize in possiblebatch:
            for split in splitnum:
                for model_type in models:
                    for dicttype in trainortest:
                        # targetfile = "BERT_abalated_b16l5_SGMEMES_split0_5e-05_16epoch_29_predictiondict_test_abalated_b16l5_SGMEMES.json"
                        # targetfile = "BERT_abalated_b16l5_SGMEMES_split0_5e-05_16epoch_29_predictiondict_train_abalated_b16l5_SGMEMES.json"
                        
                        
                        targetfile_mainname = model_type+"_b"+str(batchsize)+"l"+str(lr)+"_SGMEMES"
                        targetfile = "BERT_"+targetfile_mainname+"_split"+str(split)+"_"+str(lr)+"e-05_"+str(batchsize)+"epoch_29_predictiondict_" + dicttype+"_"+targetfile_mainname+".json"
                        
                        # targetfile = "BERT_normal_b16l5_SGMEMES_split0_5e-05_16epoch_29_predictiondict_test_normal_b16l5_SGMEMES.json"
                        # targetfile = "BERT_normal_b16l5_SGMEMES_split0_5e-05_16epoch_29_predictiondict_train_normal_b16l5_SGMEMES.json"

                        if not os.path.exists(targetfile):
                            continue

                        print(targetfile)
                        computed_results_dict["Learning Rate"].append(lr)
                        computed_results_dict["Batch Size"].append(batchsize)
                        computed_results_dict["Model Type"].append(model_type)
                        computed_results_dict["split"].append(split)
                        computed_results_dict["Type"].append(dicttype)


                        with open(targetfile,"r",encoding="utf-8") as openedfile:
                            loaded_results = json.load(openedfile)
                        
                        allowed_images = []
                        for sourceimages in  loaded_results:
                            allowed_images.append(sourceimages)
                        
                        
                        dataset_main = dual_dataset("SGMEMES_dataset_processed_final.json",target_tokeniser=["bert"],verbose=False,approved_images=allowed_images)
                        totalentities = 0
                        pureentities = 0
                        correctentities = 0 
                        total_relation_pairs = 0
                        relation_classwisedict =  {}
                        relation_classwisef1dict = {}
                        mean_average_precision_dict = {"total":0,"average_precision_list":[]}
                        for idx in range(9):
                            relation_classwisef1dict[idx] = {"FP":0,"TP":0,"TN":0,"FN":0}
                            relation_classwisedict[idx] = {"correct":0,"wrong":0,"total":0}
                        single_recorder = {"total":0,"correct":0,"internaldict":{}}
                        entity_f1_dict = {"FP":0,"TP":0,"TN":0,"FN":0}
                        for data_out in dataset_main:
                            if not data_out["source_image"] in loaded_results:
                                continue
                            imagefilename = data_out["source_image"]
                            
                            
                            
                            
                            
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
                                mean_average_precision_dict["average_precision_list"].append(summer)
                                mean_average_precision_dict["total"] = mean_average_precision_dict["total"] + 1
                                
                                
                                

                                for idx in range(len(relations[relationpair]["Prediction (Logits)"])):
                                    if relations[relationpair]["Prediction (Logits)"][idx]>relationthreshold:
                                        selected_targets.append(idx)
                                    
                                for idx in range(9):
                                    if not idx in single_recorder["internaldict"]:
                                        single_recorder["internaldict"][idx] = {"FP":0,"TP":0,"TN":0,"FN":0}
                                
                                single_recorder["total"] = single_recorder["total"] + 1
                                total_relation_pairs+=1
                                
                                
                                for selected in selected_targets:

                                    if selected in correct_targets: # True positive
                                        relation_classwisef1dict[selected]["TP"] = relation_classwisef1dict[selected]["TP"] + 1
                                        single_recorder["internaldict"][selected]["TP"] = single_recorder["internaldict"][selected]["TP"] + 1
                                        
                                        
                                    elif not selected in correct_targets: # false positive
                                        relation_classwisef1dict[selected]["FP"] = relation_classwisef1dict[selected]["FP"] + 1
                                        single_recorder["internaldict"][selected]["FP"] = single_recorder["internaldict"][selected]["FP"] + 1
                                        # print(imagefilename, "FP",archetype)
                                        # print(data_out["text_locs"])
                                        # input()
                                        
                                for target in correct_targets:
                                    if not target in selected_targets: # False Negative
                                        relation_classwisef1dict[target]["FN"] = relation_classwisef1dict[target]["FN"] + 1
                                        single_recorder["internaldict"][target]["FN"] = single_recorder["internaldict"][target]["FN"] + 1
                                        # print(imagefilename, "FN",archetype)
                                        # print(data_out["text_locs"])
                                        # input()
                                        
                                for idx in range(9):
                                        
                                    if not idx in correct_targets and not idx in selected_targets:
                                        relation_classwisef1dict[idx]["TN"] = relation_classwisef1dict[idx]["TN"] + 1 
                                        single_recorder["internaldict"][idx]["TN"] = single_recorder["internaldict"][idx]["TN"] + 1
                                
                                
                                for correct in correct_targets:
                                    if correct in selected_targets:
                                        relation_classwisedict[correct]["correct"] = relation_classwisedict[correct]["correct"] + 1
                                        single_recorder["correct"] = single_recorder["correct"] + 1
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
                                print("Incorrect Entity in:",imagefilename)
                                print()
                                
                                
                        print(entity_f1_dict)
                        if entity_f1_dict["TP"] + entity_f1_dict["FP"]==0:
                            precision = 0
                        else:
                            precision = entity_f1_dict["TP"]/ (entity_f1_dict["TP"] + entity_f1_dict["FP"])
                        if entity_f1_dict["TP"] + entity_f1_dict["FN"]==0:
                            recall = 0
                        else:
                            recall = entity_f1_dict["TP"]/ (entity_f1_dict["TP"] + entity_f1_dict["FN"])
                        
                        if precision+recall==0:
                            f1_score = 0
                        else:
                            f1_score = 2*precision*recall /(precision + recall)
                        print("Entity F1 Score",f1_score)
                        print("Entity Precision",precision)
                        print("Entity Recall",recall)
                            
                        print("Total entities:",totalentities)
                        print("Correct entities:",pureentities)
                        # print("Correct(onlyflag) entities:",correctentities)
                        print("Correct entities Percentage:",round(pureentities/totalentities,4)*100,"%")
                        # print("Correct(onlyflag) entities Percentage:",round(correctentities/totalentities,4)*100,"%")    
                        print("Total Relation Pairs:",total_relation_pairs)
                        
                        computed_results_dict["Total Entities"].append(totalentities)
                        computed_results_dict["Correct Entities"].append(pureentities)
                        computed_results_dict["Entity Correct Percentage"].append(str(round(pureentities/totalentities,4)*100)+"%")
                        
                        
                        
                        flipped_actiondict = {i:k for k,i in remapped_actionables_dict.items()}
                        print("-"*30)
                        macrof1summer = 0
                        counter = 0
                        print(relation_classwisedict)
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
                        
                            
                        totalTP = 0
                        totalFP = 0
                        totalFN = 0
                        totalTN = 0
                        for item in list(single_recorder["internaldict"].keys()):
                            totalTP += single_recorder["internaldict"][item]["TP"]
                            totalFP += single_recorder["internaldict"][item]["FP"]
                            totalFN += single_recorder["internaldict"][item]["FN"]
                            totalTN += single_recorder["internaldict"][item]["TN"]
                            
                            
                            if single_recorder["internaldict"][item]["TP"] + single_recorder["internaldict"][item]["FP"] ==0:
                                precision = 0
                            else:
                                precision = single_recorder["internaldict"][item]["TP"]/ (single_recorder["internaldict"][item]["TP"]+single_recorder["internaldict"][item]["FP"])
                            
                            if single_recorder["internaldict"][item]["TP"] + single_recorder["internaldict"][item]["FN"] ==0:
                                recall = 0 
                            else:
                                recall = single_recorder["internaldict"][item]["TP"]/(single_recorder["internaldict"][item]["TP"] + single_recorder["internaldict"][item]["FN"])
                            
                            if precision+recall==0:
                                single_recorder["internaldict"][item]["f1_score"] = 0
                            else:
                                f1_score = 2*precision*recall /(precision + recall)
                                single_recorder["internaldict"][item]["f1_score"] = f1_score
                                
                                

                        
                        
                        single_recorder["TP"] = totalTP
                        computed_results_dict["Relation Micro TP"].append(totalTP)
                        single_recorder["TN"] = totalTN
                        computed_results_dict["Relation Micro TN"].append(totalTN)
                        single_recorder["FP"] = totalFP
                        computed_results_dict["Relation Micro FP"].append(totalFP)
                        single_recorder["FN"] = totalFN
                        computed_results_dict["Relation Micro FN"].append(totalFN)
                        
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
                        single_recorder["f1score"] = round(f1score,4)
                        computed_results_dict["Relation Micro Precision"].append(precision)
                        computed_results_dict["Relation Micro Recall"].append(recall)
                        computed_results_dict["Relation Micro F1"].append(round(f1score,4))
                        overall_mAP_result = sum(mean_average_precision_dict["average_precision_list"])/mean_average_precision_dict["total"]
                        print("Mean Average Precision",overall_mAP_result)
                        # computed_results_dict["relation_mAP"] = overall_mAP_result
                        
                        print("-"*50)
                        print("\n\n")
                        
                        masscorrect = 0
                        masstotal = 0
                        print("overall_relationship accuracy:",round(single_recorder["correct"]/single_recorder["total"]*100,2))
                        masstotal+= single_recorder["total"]
                        computed_results_dict["Total Relation Pairs"].append(total_relation_pairs)
                        masscorrect += single_recorder["correct"]
                        computed_results_dict["Relations Correct"].append(masscorrect)
                        
                        print("Overall mass accuracy",masscorrect/masstotal*100)
                        computed_results_dict["Relation Accuracy"].append(masscorrect/masstotal*100)
                        print("\n\n\n\n")
                        print("f1 score: ",single_recorder["f1score"])
                        
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
                        
                            
                            
                        # pprint.pprint(archetype_recorder)
        
        
    with open("SGMEMES_Comparison_file.csv","w",encoding="utf-8",newline="") as comparisondumpfile:
        writer = csv.DictWriter(comparisondumpfile,fieldnames=list(computed_results_dict.keys()))
        writer.writeheader()
        for computed_targetkey in range(len(computed_results_dict[list(computed_results_dict.keys())[0]])):
            dumpdict = {}
            # print(computed_results_dict)
            for results_dict_key in computed_results_dict:
                # print(results_dict_key)
                dumpdict[results_dict_key] = computed_results_dict[results_dict_key][computed_targetkey]
            writer.writerow(dumpdict)
        