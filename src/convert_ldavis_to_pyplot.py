import json
import math
import matplotlib.pyplot as plt

def convert_ldavis_to_pyplot(jsonfile, outfile):
    # Convert image to json so we can create an svg
    with open(jsonfile, 'r') as myfile:
        data=myfile.read()

    json_data = json.loads(data)

    plt.figure(figsize=(6,6))

    x_data = json_data['mdsDat']['x']
    y_data = json_data['mdsDat']['y']
    freq_data = json_data['mdsDat']['Freq']

    x_min, x_max = min(x_data), max(x_data)
    y_min, y_max = min(y_data), max(y_data)
    x_range = x_max - x_min
    y_range = y_max - y_min

    plt.axis([x_min - x_range / 2, x_max + x_range / 2, y_min - y_range / 2, y_max + y_range / 2])
    plt.axis("equal")

    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=False)
    plt.xlabel('PC1')
    plt.ylabel('PC2')


    # Depending on the number of topics, you may need to tweak the paremeters (e.g. the size of circles be Freq/100 or Freq/200, etc)

    for i in range(len(json_data['mdsDat']['x'])):
        radius = math.log(freq_data[i] * 3) / 80
        circle = plt.Circle((x_data[i], y_data[i]), radius=radius, edgecolor='grey', 
                            facecolor='lightblue', 
                            alpha=0.5)
        plt.gca().add_artist(circle)

        plt.text(x_data[i], y_data[i], str(i + 1), ha='center', va='center')  # label topics


    plt.savefig(outfile, format = 'eps', bbox_inches='tight')


    plt.show()