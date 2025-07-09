import os
import torch
import numpy as np
from network import BaseNetwork, SiameseNetwork  # type: ignore
from seqential import SimilarityNetwork  # type: ignore
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using:', device)

    scaler = joblib.load(r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\scaler.pkl')

    labels_dir = r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\Augement_data_whole\Test\label'
    pairs_dir = r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\Augement_data_whole\Test\pairs'
    input_size = 33
    
    base_network = BaseNetwork(input_size).to(device)
    model = SiameseNetwork(base_network).to(device)
    model.load_state_dict(torch.load(r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\siamese_model_weights.pth', map_location=device))
    model.eval()

    similarity_model = SimilarityNetwork().to(device)
    similarity_model.load_state_dict(torch.load(r'C:\Users\UZEX5M4\Siamese NN_Final\New_SNN\similarity_model_weights_1.pth', map_location=device))
    similarity_model.eval()

    true_labels = []
    predicted_labels = []
    all_distances = []
    all_embeddings = []
    pairs = []
    hover_data = []
    distance_threshold = 2.0

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.npy'):
            single_label_file = os.path.join(labels_dir, label_file)
            single_pair_file = os.path.join(pairs_dir, f'pair_{label_file.split("_")[1]}')

            test_label = np.load(single_label_file, allow_pickle=True)

            pair = np.load(single_pair_file, allow_pickle=True)
            scenario_1 = np.array(pair[0], dtype=np.float32)
            scenario_2 = np.array(pair[1], dtype=np.float32)

            length_1 = scenario_1.shape[0]
            length_2 = scenario_2.shape[0]

            scenario_1 = torch.tensor(scaler.transform(scenario_1), dtype=torch.float32)
            scenario_2 = torch.tensor(scaler.transform(scenario_2), dtype=torch.float32)
            lengths1 = torch.tensor([length_1], dtype=torch.long)
            lengths2 = torch.tensor([length_2], dtype=torch.long)

            scenarios = [scenario_1, scenario_2]
            padded_scenarios = pad_sequence(scenarios, batch_first=True)
            scenario_1_padded = padded_scenarios[0].unsqueeze(0).to(device)
            scenario_2_padded = padded_scenarios[1].unsqueeze(0).to(device)

            with torch.no_grad():
                output1, output2 = model(scenario_1_padded, scenario_2_padded, lengths1.to(device), lengths2.to(device))
                euclidean_distance = F.pairwise_distance(output1, output2)
                distance_input = euclidean_distance.unsqueeze(1)
                similarity_score = similarity_model(distance_input)
               
                probability = similarity_score.item()
                predicted_label = 1 if euclidean_distance < distance_threshold  and similarity_score > 0.5 else 0
                predicted_labels.append(predicted_label)
                true_labels.append(test_label)
                all_distances.append(distance_input.item())

                all_embeddings.append(output1.cpu().numpy().flatten())
                all_embeddings.append(output2.cpu().numpy().flatten())
                pairs.append((len(all_embeddings) - 2, len(all_embeddings) - 1))
                hover_data.append({
                    "id_1": len(all_embeddings) - 2,
                    "id_2": len(all_embeddings) - 1,
                    "distance": euclidean_distance.item(),
                    "similarity_score" : probability,
                    "label": label_file 
                })

                print(f'File: {label_file}')
                print('----------------------------------')
                print(f'Euclidean Distance: {euclidean_distance.item():.4f}')
                similarity_score =probability*100
                print(f'probility: {similarity_score:.2f}%')
                print(f'Predicted: {"Similar" if predicted_label == 1 and similarity_score > 60 else "Dissimilar"}')
                print(f'Original: {"Similar" if test_label == 1 else "Dissimilar"}')
                print('***********************************')

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dissimilar', 'Similar'], yticklabels=['Dissimilar', 'Similar'])
    plt.xlabel('Predicted', fontsize = 14)
    plt.ylabel('True', fontsize = 14)
    plt.title('Confusion Matrix' ,fontsize = 14)
    plt.show()

    # PCA Plot
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(np.array(all_embeddings))

    fig, ax = plt.subplots(figsize=(10, 8))
    points = {}
    lines = {}
    distance_text = None  
    label_text = None  
    label_texts = []

    for pair, hover_info in zip(pairs, hover_data):
        id_1, id_2 = pair
        p1, p2 = reduced_embeddings[id_1], reduced_embeddings[id_2]

        point1, = ax.plot(p1[0], p1[1], 'ro', label='Scenario 1' if id_1 == 0 else "")
        points[id_1] = point1
        point2, = ax.plot(p2[0], p2[1], 'b*', label='Scenario 2' if id_2 == 1 else "")
        points[id_2] = point2

        line = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.5, visible=False)
        lines[(id_1, id_2)] = line[0]

    def on_hover(event):
        global distance_text, label_text, label_texts
        if event.inaxes == ax:
            for idx, point in points.items():
                if point.contains(event)[0]:
                    corresponding_pair = [d for d in hover_data if d['id_1'] == idx or d['id_2'] == idx][0]
                    highlight_id = corresponding_pair['id_2'] if corresponding_pair['id_1'] == idx else corresponding_pair['id_1']
                    corresponding_point = points[highlight_id]
                    corresponding_line = lines[(corresponding_pair['id_1'], corresponding_pair['id_2'])]

                    point.set_color('green')
                    corresponding_point.set_color('orange')
                    corresponding_line.set_visible(True)
                    distance = corresponding_pair['distance']
                    label = corresponding_pair['label']

                    mid_x = (reduced_embeddings[corresponding_pair['id_1']][0] + reduced_embeddings[corresponding_pair['id_2']][0]) / 2
                    mid_y = (reduced_embeddings[corresponding_pair['id_1']][1] + reduced_embeddings[corresponding_pair['id_2']][1]) / 2
                    if distance_text:
                        distance_text.remove()
                    distance_text = ax.text(mid_x, mid_y, f'{distance:.4f}', fontsize=10, color='purple', ha='center')

                    if label_text:
                        label_text.remove()
                    label_text = ax.text(mid_x, mid_y + 0.2, f'label: {label}', fontsize=10, color='black', ha='center')

                    ax.set_title(f'Distance: {distance:.4f}')
                    fig.canvas.draw_idle()
                    break
        else:
           
            for point in points.values():
                point.set_color('r' if point.get_marker() == 'o' else 'b')
            for line in lines.values():
                line.set_visible(False)
            if distance_text:
                distance_text.remove()
                distance_text = None
            if label_text:
                label_text.remove()
                label_text = None
            ax.set_title('PCA Plot of Embeddings', fontsize = 14)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Test_case_i', markerfacecolor='red', markersize=10),
                       Line2D([0], [0], marker='*', color='w', label='Test_case_j', markerfacecolor='blue', markersize=10)]
    ax.legend(handles=legend_elements, loc='best')
    ax.set_title('PCA Plot of Embeddings', fontsize = 14)
    ax.set_xlabel('Principal Component 1', fontsize =14)
    ax.set_ylabel('Principal Component 2', fontsize = 14)
    plt.grid(True)
    plt.show()

  
    theta = np.linspace(0, 2 * np.pi, len(all_distances))
    radii = all_distances
    colors = ['blue' if label == 1 else 'red' for label in predicted_labels]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
scatter = ax.scatter(theta, radii, c=colors, alpha=0.7)

ax.plot(np.linspace(0, 2 * np.pi, 100), [distance_threshold] * 100, color="green", linestyle="--", label="Margin=2(Threshold)")

ax.set_xticks([])  

annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    idx = ind["ind"][0]  
    theta_val = theta[idx]  
    r_val = radii[idx] 
    label = hover_data[idx]["label"]  

    annot.xy = (theta_val, r_val)
    annot.set_text(f"Distance: {r_val:.2f}\nLabel: {label}", fontsize =14)
    annot.get_bbox_patch().set_alpha(0.9)

def on_hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        contains, ind = scatter.contains(event)
        if contains:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_hover)

ax.set_title("Circular Scatter Plot of Distances between Dis-similar test cases with similar test case", fontsize = 14)
legend_elements = [
    Line2D([0], [0], color="green", linestyle="--", label=" Margin=2(Threshold)")
]
ax.legend(handles=legend_elements, loc="upper right")

plt.show()
if isinstance(all_distances, float):
    all_distances = [all_distances]


if len(true_labels) == len(predicted_labels) == len(all_distances):

    heatmap_data = np.array([
        true_labels,         
        predicted_labels,    
        all_distances        
    ])

    annotated_labels = np.array([
        ["similar" if label == 1 else "dissimilar" for label in true_labels],         
        ["similar" if label == 1 else "dissimilar" for label in predicted_labels],    
        [f"{dist:.2f}" for dist in all_distances]                                   
    ])

   
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        annot=annotated_labels, 
        fmt="",                 
        cmap="coolwarm",          
        cbar=True,               
        cbar_kws={"label": "Value Intensity (0 to 1)"}, 
        xticklabels=[f"Pair {i+1}" for i in range(len(true_labels))],
        yticklabels=["True Label", "Prediction", "Distance"]
    )
    plt.title("Comparison of True Labels, Predicted Labels, and Distances", fontsize=14)
    plt.xlabel("Pair Index", fontsize=14)
    plt.ylabel("Metrics", fontsize=14)
    plt.show()

num_scenarios = max(max(pair["id_1"], pair["id_2"]) for pair in hover_data) + 1

similarity_matrix = np.zeros((num_scenarios, num_scenarios))
border_mask = np.zeros_like(similarity_matrix, dtype=bool)

for pair in hover_data:
    id_1 = pair["id_1"]
    id_2 = pair["id_2"]
    similarity_score = pair["similarity_score"]

    similarity_matrix[id_1, id_2] = similarity_score*100
    similarity_matrix[id_2, id_1] = similarity_score*100  # Ensure symmetry

    border_mask[id_1, id_2] = True
    border_mask[id_2, id_1] = True

scenario_labels = [f"Test Case {i+1}" for i in range(num_scenarios)]

plt.figure(figsize=(12, 10))

sns.heatmap(
    similarity_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=scenario_labels,
    yticklabels=scenario_labels,
    cbar_kws={'label': 'Similarity Score'},
    linewidths=0, 
    linecolor='none'
)

for i in range(num_scenarios):
    for j in range(num_scenarios):
        if border_mask[i, j]:  
            plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=1.5))

plt.title("Heatmap of similarity score between test cases in (%)")
# plt.xlabel("Scenario (j)", fontsize = 14)
# plt.ylabel("Scenario (i)", fontsize = 14)
plt.show()