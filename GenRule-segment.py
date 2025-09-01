import numpy as np
import pandas as pd
import os
from collections import defaultdict

def parse_line(line):
    i = 0
    num1 = 0
    num_str = ''
    while line[i] != ',':
        if line[i] == '[':
            pass
        else:
            num_str = num_str + line[i]
        i += 1
    num1 = int(num_str)
    addr1 = ''
    l1 = 0
    while line[i] != ']':
        if line[i] == "'":
            l1 = 1
        elif l1 == 1:
            addr1 += line[i]
        i += 1

    num2 = 0
    l2 = 0
    num_str = ''
    while line[i] != ',':
        if line[i] == '[':
            l2 = 1
        elif l2 ==1 :
            num_str = num_str + line[i]
        i += 1
    num2 = int(num_str)
    l2 = 0
    addr2 = ''
    while line[i] != ']':
        if line[i] == "'":
            l2 = 1
        elif l2 == 1:
            addr2 += line[i]
        i += 1
    return [num1,addr1,num2,addr2]
def head(name):
    file.write("rule "+name+":\n")
def let_in(var,val):
    file.write("let\n")
    for l in range(0,len(var)):
        file.write(var[l]+"="+val[l]+"\n")
    file.write("in\n")
def action_left(action):
    file.write("[")
    for l in range(0,len(action)):
        file.write(action[l])
        if l != (len(action)-1):
            file.write(",")
    file.write("]\n")
def action_right(action):
    file.write("[")
    for l in range(0,len(action)):
        file.write(action[l])
        if l != (len(action)-1):
            file.write(",")
    file.write("]\n")
def action_middle(action):
    file.write("--[")
    for l in range(0,len(action)):
        file.write(action[l])
        if l != (len(action)-1):
            file.write(",")
    file.write("]->\n")
def rule_gen(name,var,val,act_l,act_m,act_r):
    head(name)
    if len(var) != 0:
        let_in(var,val)
    action_left(act_l)
    action_middle(act_m)
    action_right(act_r)
    file.write("\n")
tag = defaultdict(lambda: "UNKNOW")
tag.update({1:'CONNECT',2:'CONNACK',3:'PUBLISH',4:'PUBACK',5:'PUBREC',6:'PUBREL',
       7:'PUBCOMP',8:'SUBSCRIBE',9:'SUBACK',10:'UNSUBSCRIBE',11:'UNSUBACK',
       12:'PINGREQ',13:'PINGRESP',14:'DISCONNECT'})
fun = defaultdict(lambda: "UN_Know_State")
fun.update({1:'Conn_Request',2:'Conn_Msg_Ack',3:'Pub_Msg',4:'Msg_Pub_Rcv_Ack1',
       5:'Pub_Rcv1',6:'Pub_Rel2',7:'Pub_Comp3',8:'Sub_Req',9:'Sub_Msg_Ack',
       10:'UN_Sub_Req',11:'UN_Sub_Ack',12:'Heart_Req',13:'Heart_Resp',14:'Dis_Conn'})

tr = "Tr_"
st = "St_"

sp = """
lemma PS1:
\"
	All a b c d e f #i #j. #j<#i & PUBLISH(a,b,c,d)@i&SUBSCRIBE(e,d,f,b)@j & (not (d=b)) ==> Ex s v #k. #j<#i & #j<#k & #k<#i & SUBACK(s,b,v,d)@k
\"

lemma PS2:
\"
	//Publish occur bofore agents must build connect
All a b c d e f #i #k. #i<#k & DISCONNECT(a,b,c,d)@i & CONNACK(e,d,f,b)@k ==>
			(not Ex q w #j. #i<#j & #j<#k & PUBLISH(q,d,w,b)@j) 
			| (not Ex q w #j. #i<#j & #j<#k & PUBLISH(q,b,w,d)@j)
			| (Ex #j. #j<#i & Ini(a,b)@j)
			| (Ex #j. #j=#i & Ini(a,b)@j)

\"

lemma PS3:
\"
All a b c d e f #i #l. #i<#l & UNSUBSCRIBE(a,b,c,d)@i & UNSUBACK(e,d,f,b)@l ==>
			(not Ex e f #j . #i<#j & PUBLISH(e,d,f,b)@j) | (Ex #j. #j<#i & Ini(a,b)@j) | (Ex #j. #j=#i & Ini(a,b)@j)

		
\"

lemma PS4:
\"
	All a b c d e g #i #l. #l<#i & PUBACK(a,b,c,d)@i & PUBLISH(e,d,g,b)@l ==>
										(Ex z x v#j. PUBLISH(z,x,v,b)@j )
\"

lemma PS5:
\"
	All a b c d f g #k #i. #k<#i&PUBCOMP(a,b,c,d)@i&PUBLISH(f,d,g,b)@k==> 
	(Ex q e s r #j #l. #l<#i&#j<#l&#k<#j&PUBREC(e,b,q,d)@j&PUBREL(s,d,r,b)@l)
											|
											(Ex #j. #j<#i & Ini(a,b)@j)
											|
											(Ex #j. #j=#i & Ini(a,b)@j)
\"

lemma PS6:
\"
	All a b c d #i #j. #i<#j & PINGREQ(a,b,c,d)@i & PINGRESP(c,d,a,b)@j ==>
			(not Ex #k. #k<#i & DISCONNECT(a,b,c,d)@k)
			|
			(Ex e f #k. #k<#i & PUBLISH(a,b,e,f)@k)
			|
			(Ex e f #k. #k<#i & SUBSCRIBE(a,b,e,f)@k)
			|
			(Ex e f #k. #k<#i & UNSUBSCRIBE(a,b,e,f)@k)
			|
			(Ex e f #k. #k<#i & PUBLISH(e,f,a,b)@k)
			|
			(Ex e f #k. #k<#i & SUBSCRIBE(e,f,a,b)@k)
			|
			(Ex e f #k. #k<#i & UNSUBSCRIBE(e,f,a,b)@k) 
			|
			(Ex #j. #j<#i & Ini(a,b)@j)
			|
			(Ex #j. #j=#i & Ini(a,b)@j)
\" 
"""


rule_set = []
state_ini = []
buffer_addr = []
seg = list((100,500,1000,2000,5000))
#seg = list((100,200,300,400,500,600))
for cc in range(0,len(seg)):
    with open('output_pre_'+str(seg[cc])+'.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            transition = parse_line(line)
            if transition[1] not in buffer_addr:
                buffer_addr.append(transition[1])
                state_ini.append([transition[0],transition[1]])
            if [transition[0],transition[1],transition[2],transition[3]] not in rule_set:
                rule_set.append(transition)
                #print('[',transition[0],transition[1],']',"--[",transition[2],"]-->",'[',transition[2],transition[3],']')
            else:
                print("Repeat")
        file.close()
        print(rule_set)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'MQTT_pre_'+str(seg[cc])+'.spthy'
    file_path = os.path.join(current_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    with open('MQTT_pre_'+str(seg[cc])+'.spthy', 'w', encoding='utf-8') as file:
        file.write("theory MQTT_pre_"+str(seg[cc])+"\n"+"begin")
        file.write("builtins: hashing, symmetric-encryption, asymmetric-encryption, signing,diffie-hellman"+"\n")
        file.write("section{*MQTT State_"+str(seg[cc])+"*}"+"\n")
        action_l = []
        action_m = []
        action_r = []
        let_var = []
        let_val = []
        c_t = 0
        for re in enumerate(state_ini):
            var = "col1"
            let_var.append(var)
            let_val.append("'" + str(re[1][0]) + "'")


            var1 = "addr1"
            let_var.append(var1)
            let_val.append("'" + re[1][1].replace('.', '_') + "'")

            var3 = "cr1"
            let_var.append(var3)
            let_val.append("'" + fun[re[1][0]] + "'")


            state1 = "!" + st + str(re[1][0]) + "_" + re[1][1].replace('.', '_') + "("  + var3 + ","  + var1 + ")"
            action_r.append(state1)
            action_m.append("Ini"+"("+var+"," +var1+"),"+str(tag[re[1][0]])+"("+var +"," +var1+","+var+","+var1+")")
            name = "Ini_" + str(c_t)
            rule_gen(name, let_var, let_val, action_l, action_m, action_r)
            c_t += 1
            action_l.clear()
            action_m.clear()
            action_r.clear()
            let_var.clear()
            let_val.clear()

        action_l = []
        action_m = []
        action_r = []
        let_var = []
        let_val = []
        c_t = 0
        for re in enumerate(rule_set):

            var = "col1"
            let_var.append(var)
            let_val.append("'" + str(re[1][0]) + "'")

            var0 = "col2"
            let_var.append(var0)
            let_val.append("'" + str(re[1][2]) + "'")

            var1 = "addr1"
            let_var.append(var1)
            let_val.append("'" + re[1][1].replace('.', '_') + "'")

            var2 = "addr2"
            let_var.append(var2)
            let_val.append("'" + re[1][3].replace('.', '_') + "'")

            var3 = "cr1"
            let_var.append(var3)
            let_val.append("'" + fun[re[1][0]] + "'")

            var4 = "cr2"
            let_var.append(var4)

            let_val.append("'" + fun[re[1][2]] + "'")

            state1 = "!" + st + str(re[1][0]) + "_" + re[1][1].replace('.', '_') + "("  + var3 + ","  + var1 + ")"
            state2 = "!" + st + str(re[1][2]) + "_" + re[1][3].replace('.', '_') + "("  + var4 + ","  + var2 + ")"
            action_l.append(state1)
            action_r.append(state2)

            action_m.append(str(tag[re[1][2]]) + "("  + var + ","  + var1 + "," + var0 + "," + var2 +")")
            name = tr + str(c_t)
            rule_gen(name, let_var, let_val, action_l, action_m, action_r)
            c_t += 1
            action_l.clear()
            action_m.clear()
            action_r.clear()
            let_var.clear()
            let_val.clear()
            print(cc)

        file.write(sp+"\n")
        file.write("\n" + "end" + "\n")

    #Origin
    rule_set = []
    state_ini = []
    buffer_addr = []
    with open('output_real_'+str(seg[cc])+'.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            transition = parse_line(line)
            if transition[1] not in buffer_addr:
                buffer_addr.append(transition[1])
                state_ini.append([transition[0],transition[1]])
            if [transition[0],transition[1],transition[2],transition[3]] not in rule_set:
                rule_set.append(transition)
                #print('[',transition[0],transition[1],']',"--[",transition[2],"]-->",'[',transition[2],transition[3],']')
            else:
                print("Repeat")
        file.close()
        print(rule_set)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'MQTT_real_'+str(seg[cc])+'.spthy'
    file_path = os.path.join(current_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    with open('MQTT_real_'+str(seg[cc])+'.spthy', 'w', encoding='utf-8') as file:
        file.write("theory MQTT_real_"+str(seg[cc])+"\n"+"begin")
        file.write("builtins: hashing, symmetric-encryption, asymmetric-encryption, signing,diffie-hellman"+"\n")
        file.write("section{*MQTT State_"+str(seg[cc])+"*}"+"\n")
        action_l = []
        action_m = []
        action_r = []
        let_var = []
        let_val = []
        c_t = 0
        for re in enumerate(state_ini):
            var = "col1"
            let_var.append(var)
            let_val.append("'" + str(re[1][0]) + "'")


            var1 = "addr1"
            let_var.append(var1)
            let_val.append("'" + re[1][1].replace('.', '_') + "'")

            var3 = "cr1"
            let_var.append(var3)
            let_val.append("'" + fun[re[1][0]] + "'")


            state1 = "!" + st + str(re[1][0]) + "_" + re[1][1].replace('.', '_') + "("  + var3 + ","  + var1 + ")"
            action_r.append(state1)
            action_m.append("Ini" + "(" + var + ","  + var1 +")")
            name = "Ini_" + str(c_t)
            rule_gen(name, let_var, let_val, action_l, action_m, action_r)
            c_t += 1
            action_l.clear()
            action_m.clear()
            action_r.clear()
            let_var.clear()
            let_val.clear()

        action_l = []
        action_m = []
        action_r = []
        let_var = []
        let_val = []
        c_t = 0
        for re in enumerate(rule_set):

            var = "col1"
            let_var.append(var)
            let_val.append("'" + str(re[1][0]) + "'")

            var0 = "col2"
            let_var.append(var0)
            let_val.append("'" + str(re[1][2]) + "'")

            var1 = "addr1"
            let_var.append(var1)
            let_val.append("'" + re[1][1].replace('.', '_') + "'")

            var2 = "addr2"
            let_var.append(var2)
            let_val.append("'" + re[1][3].replace('.', '_') + "'")

            var3 = "cr1"
            let_var.append(var3)
            let_val.append("'" + fun[re[1][0]] + "'")

            var4 = "cr2"
            let_var.append(var4)
            let_val.append("'" + fun[re[1][2]] + "'")

            state1 = "!" + st + str(re[1][0]) + "_" + re[1][1].replace('.', '_') + "("  + var3 + ","  + var1 + ")"
            state2 = "!" + st + str(re[1][2]) + "_" + re[1][3].replace('.', '_') + "("  + var4 + ","  + var2 + ")"
            action_l.append(state1)
            action_r.append(state2)

            action_m.append(str(tag[re[1][2]]) + "("  + var + ","  + var1 + "," + var0 + "," + var2 +")")
            name = tr + str(c_t)
            rule_gen(name, let_var, let_val, action_l, action_m, action_r)
            c_t += 1
            action_l.clear()
            action_m.clear()
            action_r.clear()
            let_var.clear()
            let_val.clear()
        file.write(sp + "\n")
        file.write("\n" + "end" + "\n")

