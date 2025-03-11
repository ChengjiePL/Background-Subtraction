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
alpha = 1.5;
beta = 5;

% Crear una figura para mostrar los resultados
fig4 = figure('Name', 'Comparación de Modelos', 'NumberTitle', 'off');

% Usar la misma imagen para ambos modelos (para una comparación justa)
selected_image_index = 50; % O usa el mismo índice que en la tarea 3 para comparar
current_image = images_gray(:,:,selected_image_index);

% Modelo simple (de Tarea 3)
difference = abs(double(current_image) - median_image);
binary_mask_simple = difference > thr;

% Modelo elaborado (Tarea 4)
diff = abs(double(current_image) - median_image);
threshold_matrix = alpha * std_image + beta;
binary_mask_elaborate = diff > threshold_matrix;

% Apply morphological operations to clean up results
se1 = strel('disk', 2); % Small structuring element for noise removal
se2 = strel('disk', 4); % Larger structuring element for closing holes

% Clean up both masks
binary_mask_simple = imopen(binary_mask_simple, se1);
binary_mask_simple = imclose(binary_mask_simple, se2);

binary_mask_elaborate = imopen(binary_mask_elaborate, se1);
binary_mask_elaborate = imclose(binary_mask_elaborate, se2);

% Mostrar resultados para comparación
subplot(2,2,1); imshow(current_image); title('Imagen Original');
subplot(2,2,3); imshow(binary_mask_simple); title('Modelo Simple (thr)');
subplot(2,2,4); imshow(binary_mask_elaborate); title('Modelo Elaborado (α*σ+β)');

uiwait(fig4); % Wait for user to close the figure

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
