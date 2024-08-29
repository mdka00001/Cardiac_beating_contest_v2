import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as find_peaks

class CardiacBeatingBase:


        
    def draw_flow(img, flow, grid_size=128, spacing=0):

        """
        Draw frames per video with eucledian norm of vector displacement in each grid
        input:
            img: input frame.
            flow: ortical flow data.
            grid_size: size of every grid in the analysis.
            spacing: distance between each grid.
        output:
            img: processed frame.
            norms: eucledian norm of each frame.
        """
        h, w = img.shape[:2]
        num_grids_x = (w - 2 * spacing) // grid_size
        num_grids_y = (h - 2 * spacing) // grid_size

        norms = []  # Store Euclidean norms for each grid
        
        for x in range(num_grids_x):
            for y in range(num_grids_y):
                x_start = spacing + x * grid_size
                y_start = spacing + y * grid_size
                
                # Calculate the average flow in the grid
                grid_flow = flow[y_start:y_start + grid_size, x_start:x_start + grid_size]
                avg_flow = np.mean(grid_flow, axis=(0, 1))
                #print(avg_flow)
                # Compute Euclidean norm of the displacement vector
                norm = np.linalg.norm(avg_flow)
                #print(norm)
                norms.append(norm)

                # Draw arrow for the flow vector
                pt1 = (x_start + grid_size // 2, y_start + grid_size // 2)
                pt2 = (int(pt1[0] + avg_flow[0] * 5), int(pt1[1] + avg_flow[1] * 5))  # Scale for visibility
                cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), 2, tipLength=0.5)

                # Optionally, display the norm value on the grid (comment out if not needed)
                cv2.putText(img, f"{norm:.2f}", (x_start, y_start + grid_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        #print("done")

        return img, norms


    def save_optical_flow_video(frames, framerate, output_filename=None, grid_size=128, spacing=0):
        # Initialize video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
        out = cv2.VideoWriter(output_filename, fourcc, framerate, (width, height))

        all_norms = []  # To store norms for all frames

        

        # Process each pair of frames
        for i in range(1, len(frames)):
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            
            #curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            

            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 7, 1.5, 0)
        
            
            # Draw flow vectors on the original frame and compute norms
            frame_with_flow, norms = CardiacBeatingBase.draw_flow(frames[i].copy(), flow, grid_size, spacing)
            all_norms.append(norms)

            # Write the frame to the video
            out.write(frame_with_flow)


        if output_filename != None:
        
            out.release()
            print(f"Video saved as {output_filename}")
        return all_norms


    def get_grid_clusters(all_norms, image_file_name):

        min_vals = np.min(all_norms, axis=0)
        max_vals = np.max(all_norms, axis=0)

        normalized_data = (all_norms - min_vals) / (max_vals - min_vals)

        grid_shape = (5, 10)
        # Step 1: Calculate Integrated Displacement for Each Grid
        integrated_displacement = np.max(all_norms, axis=0)

        # Step 2: Reshape the data for GMM
        data_for_gmm = integrated_displacement.reshape(-1, 1)
        print(np.mean(data_for_gmm), np.std(data_for_gmm))

        # Step 3: Apply Gaussian Mixture Model (GMM)
        n_components = 2  # The number of clusters you expect; adjust this as necessary
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data_for_gmm)

        # Step 4: Predict cluster labels
        cluster_labels = gmm.predict(data_for_gmm)

        # Step 5: Reshape the cluster labels back to the grid layout
        cluster_labels_reshaped = cluster_labels.reshape(grid_shape)

        # Step 6: Identify the signal group (the cluster with the highest mean)
        signal_group = np.argmax(gmm.means_)

        # Step 7: Extract grid indices that belong to the signal group
        signal_group_indices = np.argwhere(cluster_labels == signal_group)

        # If you want to access the normalized data for the signal group grids
        signal_group_data = normalized_data[:, signal_group_indices.flatten()]



        # Optional: Plotting the signal group grids on the original layout
        plt.imshow(cluster_labels_reshaped, cmap='viridis')
        plt.colorbar(label='Cluster Label')
        plt.scatter(signal_group_indices[:, 0] % grid_shape[1], signal_group_indices[:, 0] // grid_shape[1], color='red')
        plt.title('Signal Group Grids')
        plt.savefig(fr"{image_file_name}_grids.png")
        plt.clf()

        # Initialize an empty list to store the data for Cluster 1
        cluster_1_data = []

        # Iterate over the indices in Cluster 1 and extract the data
        for flat_index in signal_group_indices.flatten():
            # Directly use the flat_index to extract data for this grid across all frames
            cluster_1_data.append(normalized_data[:, flat_index])

        # Convert the list to a NumPy array for easier manipulation
        cluster_1_data = np.array(cluster_1_data)


        return cluster_1_data

        
    def get_relative_displacement_graph(cluster_1_data, frame_rate, image_file_name):

        smoothed_data = gaussian_filter1d(np.mean(cluster_1_data, axis=0), sigma=2)

        time_axis = np.linspace(0, len(smoothed_data) / frame_rate, len(smoothed_data))
        peaks, properties = find_peaks(smoothed_data, prominence=0.1, distance= 4)

        plt.plot(time_axis, smoothed_data, color='g', label='Mean Intensity')
        plt.plot(time_axis[peaks], smoothed_data[peaks], 'x')
        plt.title('Green Pixel Intensity Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Intensity')
        plt.legend()
        plt.savefig(fr"{image_file_name}_waveplot.png")
        plt.clf()

        return smoothed_data, time_axis, peaks


    def get_BPM(time_axis, peaks):

        peak_times = time_axis[peaks]  # Extract the time values at the detected peaks
        inter_beat_intervals = np.diff(peak_times)  # Calculate the differences between consecutive peak times

        # Step 3: Convert time intervals to frequency in BPM
        frequencies_bpm = 60 / inter_beat_intervals  # Convert intervals to BPM

        # Step 4: Calculate the average frequency in BPM
        average_bpm = np.mean(frequencies_bpm)

        return average_bpm

