% TASCA 1 -  Carregar el dataset i dividir-lo en dues parts iguals
dataset = '../highway/input/';

files = dir(fullfile(dataset, 'in001*.jpg'));
files = natsortfiles({files.name});

% Dividim el dataset en dues parts iguals
train_files = files(1:150);
test_files = files(151:end);

disp('✅ Dataset carregat i dividit en dues parts iguals');

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

% Display the median image
figure;
subplot(1,2,1);
imshow(uint8(median_image));
title('Median Image');

% Display the standard deviation image
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
for i = 1:size(images_gray, 3)
	if ~ishandle(fig)
		break;
	end

	% Obtener la imagen actual
	current_image = images_gray(:,:,i);

	% Restar el modelo de fondo (mediana)
	difference = abs(double(current_image) - median_image);

	% Crear máscara binaria: 0 para fondo, 1 para movimiento
	binary_mask = difference > thr;

	% Mostrar resultados para esta imagen
	subplot(1,3,1); imshow(current_image); title('Imagen Original');
	subplot(1,3,2); imshow(difference, []); title('Diferencia con el Fondo');
	subplot(1,3,3); imshow(binary_mask); title('Segmentación de Coches');

	pause(0.5); % Pausa para poder ver cada imagen
	drawnow; % Asegura que la figura se actualice
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tasca 4 - Segmentar cotxes amb un model més elaborat
fprintf('Tasca 4 - Segmentación de coches con un modelo más elaborado\n');

% Definir parámetros alfa y beta
alpha = 2;  % Multiplicador para la desviación estándar
beta = 10;    % Término constante adicional

% Crear una figura para mostrar los resultados
gcf = figure('Name', 'Modelo Elaborado', 'NumberTitle', 'off');

% Procesar cada imagen del conjunto
for i = 1:size(images_gray, 3)
	if ~ishandle(gcf)
		break;
	end

	% Obtener la imagen actual
	current_image = images_gray(:,:,i);

	% Calcular la diferencia absoluta con la mediana
	diff = abs(double(current_image) - median_image);

	% Aplicar el modelo más elaborado: P(i,j) = 1 si |I(i,j) - μ(i,j)| > α*σ(i,j) + β
	threshold_matrix = alpha * std_image + beta;
	binary_mask = diff > threshold_matrix;

	% Mostrar resultados
	subplot(2,2,1); imshow(current_image); title('Imagen Original');
	subplot(2,2,2); imshow(diff, []); title('Diferencia con el Fondo');
	subplot(2,2,3); imshow(binary_mask); title('Segmentación Final');

	pause(0.5);
	drawnow;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tasca 5 - Grabar un vídeo amb els resultats
fprintf('Tasca 5 - Grabación de vídeo con los resultados\n');

% Definir parámetros para la segmentación
alpha = 2;  % Usar los mismos parámetros de la Tasca 4
beta = 10;

% Crear un objeto VideoWriter para guardar el vídeo
video_filename = 'segmentacion_coches.avi';
video = VideoWriter(video_filename);
video.FrameRate = 10;  % 10 frames por segundo
open(video);

% Crear un elemento estructurante para operaciones morfológicas
se_close = strel('disk', 5);    % Para cerrar huecos pequeños
se_open = strel('disk', 3);     % Para eliminar ruido pequeño

fprintf('Procesando imágenes y generando vídeo...\n');

% Procesar cada imagen y añadirla al vídeo
for i = 1:size(images_gray, 3)
	% Obtener la imagen actual
	current_image = images_gray(:,:,i);

	% Calcular la diferencia absoluta con la mediana
	diff = abs(double(current_image) - median_image);

	% Aplicar el modelo para segmentación
	threshold_matrix = alpha * std_image + beta;
	binary_mask = diff > threshold_matrix;

	% Aplicar operaciones morfológicas para mejorar la segmentación
	% 1. Closing para cerrar huecos pequeños dentro de los coches
	mask_closed = imclose(binary_mask, se_close);

	% 2. Opening para eliminar pequeños puntos de ruido
	mask_processed = imopen(mask_closed, se_open);

	% Crear una imagen a color para el vídeo
	result_frame = cat(3, current_image, current_image, current_image); % Convertir a RGB

	% Marcar los coches detectados con un color rojo semitransparente
	red_mask = zeros(size(current_image), 'uint8');
	red_mask(mask_processed) = 255;
	result_frame(:,:,1) = min(255, result_frame(:,:,1) + red_mask);

	% Añadir frame al vídeo
	writeVideo(video, result_frame);

	% Mostrar progreso
	if mod(i, 10) == 0
		fprintf('Procesada imagen %d de %d\n', i, size(images_gray, 3));
	end
end

% Cerrar el vídeo
close(video);
fprintf('Vídeo guardado como: %s\n', video_filename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tasca 6 - Avalua la bondat dels teus resultats
fprintf('Tasca 6 - Evaluación de los resultados\n');

% Cargar las imágenes de groundtruth
fprintf('Cargando imágenes de groundtruth...\n');
groundtruth_dir = '../highway/groundtruth/'; % Ajusta la ruta a tu directorio de groundtruth
groundtruth_files = dir(fullfile(groundtruth_dir, 'gt001*.png')); % Ajusta la extensión si es necesario

% Definir diferentes configuraciones para evaluar
configurations = [
	% [alpha, beta, usar_morfologia]
	2.5, 10, 0;   % Configuración 1: modelo básico
	2.5, 10, 1;   % Configuración 2: con morfología
	3.0, 15, 1    % Configuración 3: parámetros ajustados con morfología
	];

% Elementos estructurantes para morfología
se_close = strel('disk', 5);
se_open = strel('disk', 3);

% Métricas para cada configuración
all_accuracies = zeros(size(configurations, 1), length(groundtruth_files));
all_precisions = zeros(size(configurations, 1), length(groundtruth_files));
all_recalls = zeros(size(configurations, 1), length(groundtruth_files));
all_f1scores = zeros(size(configurations, 1), length(groundtruth_files));

% Procesar las imágenes con cada configuración
for config_idx = 1:size(configurations, 1)
	alpha = configurations(config_idx, 1);
	beta = configurations(config_idx, 2);
	use_morphology = configurations(config_idx, 3);

	fprintf('Evaluando configuración %d: alpha=%.1f, beta=%.1f, morfología=%d\n', config_idx, alpha, beta, use_morphology);

	for img_idx = 1:length(groundtruth_files)
		% Cargar groundtruth
		gt_path = fullfile(groundtruth_dir, groundtruth_files(img_idx).name);
		gt_img = imread(gt_path);

		% Convertir groundtruth a binario (asumiendo que los vehículos tienen etiqueta > 0)
		% Ajusta esto según el formato específico de tu groundtruth
		gt_binary = gt_img > 0;

		% Obtener índice correspondiente en el conjunto de imágenes
		% Asumiendo que las imágenes de test son las últimas 150
		img_index = size(images_gray, 3) - length(groundtruth_files) + img_idx;
		current_image = images_gray(:,:,img_index);

		% Aplicar el modelo de segmentación
		diff = abs(double(current_image) - median_image);
		threshold_matrix = alpha * std_image + beta;
		binary_mask = diff > threshold_matrix;

		% Aplicar morfología si está habilitado
		if use_morphology
			binary_mask = imclose(binary_mask, se_close);
			binary_mask = imopen(binary_mask, se_open);
		end

		% Calcular métricas
		[accuracy, precision, recall, f1score] = calculateMetrics(binary_mask, gt_binary);

		all_accuracies(config_idx, img_idx) = accuracy;
		all_precisions(config_idx, img_idx) = precision;
		all_recalls(config_idx, img_idx) = recall;
		all_f1scores(config_idx, img_idx) = f1score;
	end

	% Calcular promedios para esta configuración
	fprintf('  Resultados promedio:\n');
	fprintf('  - Accuracy:  %.4f\n', mean(all_accuracies(config_idx, :)));
	fprintf('  - Precision: %.4f\n', mean(all_precisions(config_idx, :)));
	fprintf('  - Recall:    %.4f\n', mean(all_recalls(config_idx, :)));
	fprintf('  - F1 Score:  %.4f\n', mean(all_f1scores(config_idx, :)));
end

% Mostrar gráfico comparativo
figure;
bar(mean(all_accuracies, 2));
title('Comparación de Accuracy para diferentes configuraciones');
xlabel('Configuración');
ylabel('Accuracy promedio');
set(gca, 'XTickLabel', {'Básico', 'Con morfología', 'Ajustado'});
grid on;

% Función para calcular métricas
function [accuracy, precision, recall, f1score] = calculateMetrics(prediction, groundtruth)
% True Positive: predicción=1 y groundtruth=1
TP = sum(prediction(:) & groundtruth(:));
% True Negative: predicción=0 y groundtruth=0
TN = sum(~prediction(:) & ~groundtruth(:));
% False Positive: predicción=1 y groundtruth=0
FP = sum(prediction(:) & ~groundtruth(:));
% False Negative: predicción=0 y groundtruth=1
FN = sum(~prediction(:) & groundtruth(:));

% Calcular métricas
accuracy = (TP + TN) / (TP + TN + FP + FN);
precision = TP / (TP + FP + eps); % eps para evitar división por cero
recall = TP / (TP + FN + eps);
f1score = 2 * precision * recall / (precision + recall + eps);
end
