% TASCA 1 -  Carregar el dataset i dividir-lo en dues parts iguals
dataset = '../highway/input/';

files = dir(fullfile(dataset, 'in001*.jpg'));
files = sort({files.name});

% Dividim el dataset en dues parts iguals
train_files = files(1:150);
test_files = files(151:end);

disp('Dataset carregat i dividit en dues parts iguals');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TASCA 2 - Calcular la mitjana i desviació estàndard.

% Llegir la primera imatge per obtenir dimensions
img = imread(fullfile(dataset, train_files{1}));
[height, width, ~] = size(img);

% Inicialitzar un array 3D per emmagatzemar totes les imatges en escala de grisos
images_gray = zeros(height, width, length(train_files), 'uint8');

% Carregar imatges i convertir a escala de grisos
for i = 1:length(train_files)
	img = imread(fullfile(dataset, train_files{i}));
	images_gray(:,:,i) = rgb2gray(img);
end

% Calcular la mediana i desviació estàndard
median_image = median(double(images_gray), 3);
std_image = std(double(images_gray), 0, 3);

figure;
subplot(1,2,1);
imshow(uint8(median_image));
title('Median Image');

subplot(1,2,2);
std_normalized = mat2gray(std_image); % Normalize for better visualization
imshow(std_normalized);
title('Standard Deviation');
uiwait;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TASCA 3 - Segmentar cotxes restant el model de fons

fprintf('Tasca 3 - Segmentación de coches restando el modelo de fondo\n');

% Definir un umbral para determinar qué es considerado fondo
thr = 70; % Este valor puede necesitar ajustes según tus imágenes

% Crear una figura para mostrar los resultados
fig = figure('Name', 'Segmentación de coches', 'NumberTitle', 'off');

% Procesar cada imagen del conjunto
% Select which image to process (e.g., image #1)
selected_image_index = 50;

% Obtener la imagen seleccionada
current_image = images_gray(:,:,selected_image_index);

% Restar el modelo de fondo (mediana)
difference = abs(double(current_image) - median_image);

% Crear máscara binaria: 0 para fondo, 1 para movimiento
binary_mask = difference > thr;

% Mostrar resultados para esta imagen
subplot(1,2,1); imshow(current_image); title('Imagen Original');
%subplot(1,3,2); imshow(difference, []); title('Diferencia con el Fondo');
subplot(1,2,2); imshow(binary_mask); title('Segmentación de Coches');
uiwait; % Wait for user to close the figure

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tasca 4 - Segmentar cotxes amb un model més elaborat
fprintf('Tasca 4 - Segmentación de coches con un modelo más elaborado\n');

% Definir parámetros alfa y beta
alpha = 0.45;
beta = 15;

fig4 = figure('Name', 'Comparación de Modelos', 'NumberTitle', 'off');

diff = abs(double(current_image) - median_image);
threshold_matrix = alpha * std_image + beta;
binary_mask_elaborate = diff > threshold_matrix;

% Post-processing to improve binary mask
% Fill holes inside detected car regions
binary_mask_elaborate = imfill(binary_mask_elaborate, 'holes');

% Remove small noise objects
binary_mask_elaborate = bwareaopen(binary_mask_elaborate, 50);

% Morphological closing to connect nearby regions
se = strel('disk', 3);
binary_mask_elaborate = imclose(binary_mask_elaborate, se);


% Mostrar resultados para comparación
subplot(1,2,1); imshow(binary_mask); title('Modelo Simple (thr)');
subplot(1,2,2); imshow(binary_mask_elaborate); title('Modelo Elaborado (α*σ+β)');

uiwait(fig4); % Wait for user to close the figure

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tasca 5 - Grabar un vídeo amb els resultats
fprintf('Tasca 5 - Grabación de vídeo con los resultados\n');

alpha = 0.45;  % Usar los mismos parámetros de la Tasca 4
beta = 15;

% Crear un objeto VideoWriter para guardar el vídeo
video = VideoWriter('segmentacion_coches.avi');
video.FrameRate = 10;  % 10 frames por segundo
open(video);

% Crear un elemento estructurante para operaciones morfológicas
se = strel('disk', 3);

fprintf('Procesando imágenes y generando vídeo...\n');

% Procesar cada imagen y añadirla al vídeo
for i = 1:size(images_gray, 3)
	% Obtener la imagen actual
	current_image = images_gray(:,:,i);

	diff = abs(double(current_image) - median_image);
	threshold_matrix = alpha * std_image + beta;
	binary_mask = diff > threshold_matrix;
	binary_mask = imfill(binary_mask, 'holes');
	binary_mask = bwareaopen(binary_mask, 50);
	binary_mask = imclose(binary_mask, se);

	% Crear una imagen a color para el vídeo
	result_frame = cat(3, current_image, current_image, current_image); % Convertir a RGB

	% Marcar los coches detectados con un color rojo semitransparente
	red_mask = zeros(size(current_image), 'uint8');
	red_mask(binary_mask) = 255;
	result_frame(:,:,1) = min(255, result_frame(:,:,1) + red_mask);

	% Añadir frame al vídeo
	writeVideo(video, result_frame);

end

% Cerrar el vídeo
close(video);
fprintf('Vídeo guardado como: %s\n', 'segmentacion_coches.avi');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TASCA 6 - Evaluate model performance using ground truth
groundtruth_path = '../highway/groundtruth/';

% Define three cases to evaluate
cases = struct();
cases(1).name = 'Simple Model (thr=70)';
cases(1).params = struct('thr', 70, 'alpha', NaN, 'beta', NaN);
cases(2).name = 'Elaborate Model (α=0.45, β=15)';
cases(2).params = struct('thr', NaN, 'alpha', 0.45, 'beta', 15);
cases(3).name = 'Elaborate Model (α=0.6, β=20)';
cases(3).params = struct('thr', NaN, 'alpha', 0.6, 'beta', 20);

% Initialize metrics storage
num_cases = length(cases);
metrics = struct('tp', zeros(num_cases,1), 'fp', zeros(num_cases,1), ...
               'tn', zeros(num_cases,1), 'fn', zeros(num_cases,1));

% Process each test image
for i = 1:length(test_files)
    % Load test image
    test_img = imread(fullfile(dataset, test_files{i}));
    test_gray = rgb2gray(test_img);
    
    % Load ground truth (adjust filename conversion as needed)
    [~, name, ext] = fileparts(test_files{i});
	gt_name = ['gt' name(3:end) '.png']; % Use .png if ground truth is PNG
    gt_file = fullfile(groundtruth_path, gt_name);
    gt = imread(gt_file);
    gt_binary = gt == 255; % Adjust based on ground truth encoding
    
    for case_idx = 1:num_cases
        params = cases(case_idx).params;
        
        % Generate segmentation mask
        if ~isnan(params.thr)
            diff = abs(double(test_gray) - median_image);
            mask = diff > params.thr;
        else
            diff = abs(double(test_gray) - median_image);
            threshold = params.alpha * std_image + params.beta;
            mask = diff > threshold;
            mask = imfill(mask, 'holes');
            mask = bwareaopen(mask, 50);
            mask = imclose(mask, strel('disk', 3));
        end
        
        % Compute confusion matrix components
        mask = logical(mask);
        tp = sum(mask & gt_binary, 'all');
        fp = sum(mask & ~gt_binary, 'all');
        tn = sum(~mask & ~gt_binary, 'all');
        fn = sum(~mask & gt_binary, 'all');
        
        % Accumulate metrics
        metrics.tp(case_idx) = metrics.tp(case_idx) + tp;
        metrics.fp(case_idx) = metrics.fp(case_idx) + fp;
        metrics.tn(case_idx) = metrics.tn(case_idx) + tn;
        metrics.fn(case_idx) = metrics.fn(case_idx) + fn;
    end
end

% Calculate and display results
fprintf('TASCA 6 - Average Performance Metrics\n');
for case_idx = 1:num_cases
    tp = metrics.tp(case_idx);
    fp = metrics.fp(case_idx);
    tn = metrics.tn(case_idx);
    fn = metrics.fn(case_idx);
    total = tp + fp + tn + fn;
    
    accuracy = (tp + tn) / total;
    precision = tp / (tp + fp + eps); % Avoid division by zero
    recall = tp / (tp + fn + eps);
    f1 = 2 * (precision * recall) / (precision + recall + eps);
    
    fprintf('\nCase %d: %s\n', case_idx, cases(case_idx).name);
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Precision: %.4f\n', precision);
    fprintf('Recall: %.4f\n', recall);
    fprintf('F1 Score: %.4f\n', f1);
end
