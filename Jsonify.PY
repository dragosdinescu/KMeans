import json

for fileNumber in range(101, 212):
    try:
        file1 = open('participant_'+str(fileNumber)+'.txt', 'r')
        Lines = file1.readlines()

        file1.close()
        newJson = "["
        count = 0

        # Strips the newline character
        for line in Lines:
            line = line.replace("'", '"')
            if line.__contains__("Frame"):
                continue
            else:
                if line.__contains__("Head unit"):
                    line = line.replace("Head unit", '"Head Unit"')
                    if count > 0:
                        newJson += ","
                    newJson += "{" + line + ","
                elif line.__contains__("Gaze Position 3D"):
                    line = line.replace("Gaze Position 3D", '"Gaze Position 3D"')
                    newJson += line + "}"
                else:
                    line = line.replace("Left Eye", '"Left Eye"')
                    line = line.replace("Right Eye", '"Right Eye"')
                    line = line.replace("Gaze Position", '"Gaze Position"')
                    newJson += line + ","
            count += 1

        newJson += "]"
        newJson.replace(": ", ":")
        newJson.replace("}\n,{ ", "},\n{")

        d = json.loads(newJson)

        with open('participant_'+str(fileNumber)+'.json', 'w') as outf:
            json.dump(d, outf)
    except IOError:
        print("File not accessible")