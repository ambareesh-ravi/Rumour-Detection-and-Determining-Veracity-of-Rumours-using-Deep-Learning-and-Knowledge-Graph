
import json
veracity_labels={0:"true",1:"false",2:"unverified"}
sdqc_labels={0:"support",1:"deny",3:"query",2:"comment"}
def save_output(idsa, predictionsa, idsb, predictionsb, confidenceb):
    
    subtaskaenglish = {}
    subtaskbenglish = {}
    
    for i in range(len(idsa)):
      subtaskaenglish[idsa[i]]=sdqc_labels[predictionsa[i]]    

    for i in range(len(idsb)):
      subtaskbenglish[idsb[i]]=[veracity_labels[predictionsb[i]],float(confidenceb[i])]
    answer = {}
    answer['subtaskaenglish'] = subtaskaenglish
    answer['subtaskbenglish'] = subtaskbenglish
    
    answer['subtaskadanish'] = {}
    answer['subtaskbdanish'] = {}
    
    answer['subtaskarussian'] = {}
    answer['subtaskbrussian'] = {}

    #print (answer)
    #print (answer["subtaskbenglish"])
    with open("answer_try.json", 'w') as f:
        json.dump(answer, f)
