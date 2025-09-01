import numpy as np
import pandas as pd
import os

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
tag = {1:'CONNECT',2:'CONNACK',3:'PUBLISH',4:'PUBACK',5:'PUBREC',6:'PUBREL',
       7:'PUBCOMP',8:'SUBSCRIBE',9:'SUBACK',10:'UNSUBSCRIBE',11:'UNSUBACK',
       12:'PINGREQ',13:'PINGRESP',14:'DISCONNECT'}
fun = {1:'Conn_Request',2:'Conn_Msg_Ack',3:'Pub_Msg',4:'Msg_Pub_Rcv_Ack1',
       5:'Pub_Rcv1',6:'Pub_Rel2',7:'Pub_Comp3',8:'Sub_Req',9:'Sub_Msg_Ack',
       10:'UN_Sub_Req',11:'UN_Sub_Ack',12:'Heart_Req',13:'Heart_Resp',14:'Dis_Conn'}
tr = "Tr_"
st = "St_"
rule_set = []
state_ini = []
buffer_addr = []
#Predicate
with open('output_pre.txt', 'r', encoding='utf-8') as file:
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
file_name = 'MQTT_pre.spthy'
file_path = os.path.join(current_dir, file_name)
if os.path.exists(file_path):
    os.remove(file_path)
with open('MQTT_pre.spthy', 'w', encoding='utf-8') as file:
    file.write("theory MQTT_pre"+"\n"+"begin")
    file.write("builtins: hashing, symmetric-encryption, asymmetric-encryption, signing,diffie-hellman"+"\n")
    file.write("section{*MQTT State*}"+"\n")
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
    file.write("\n" + "end" + "\n")

#Origin
rule_set = []
state_ini = []
buffer_addr = []
with open('output_real.txt', 'r', encoding='utf-8') as file:
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
file_name = 'MQTT_real.spthy'
file_path = os.path.join(current_dir, file_name)
if os.path.exists(file_path):
    os.remove(file_path)
with open('MQTT_real.spthy', 'w', encoding='utf-8') as file:
    file.write("theory MQTT_real"+"\n"+"begin")
    file.write("builtins: hashing, symmetric-encryption, asymmetric-encryption, signing,diffie-hellman"+"\n")
    file.write("section{*MQTT State*}"+"\n")
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
    file.write("\n" + "end" + "\n")