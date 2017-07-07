function [X, y] = mnistTrain()
    %ファイルは/tmp/MNIST_dataに保存する
    imagefile = '/tmp/MNIST_data/train-images-idx3-ubyte';
    labelfile = '/tmp/MNIST_data/train-labels-idx1-ubyte';
    
    %画像ファイルを開く
    fid = fopen(imagefile, 'r', 'b');
    %4Byteのmagic numberをReadする
    magic_number = fread(fid, 1, 'int32');
    %Trainデータサンプル数  (60000)
    m = fread(fid, 1, 'int32');
    %画像横のピクセル数 (28)
    r = fread(fid, 1, 'int32');
    %画像縦のピクセル数(28)
    c = fread(fid, 1, 'int32');
    %[28*28 60000]Byte分をReadする
    img = fread(fid, [r*c m], 'uint8');
    fclose(fid);
    
    %Matlab/octaveの読み込み順番は列→行
    %imgは  784x60000のMatrix
    %imgを回転させる 60000 x 784 のMatrixになる
    X = img';         

    %行ごと（画像）回転させる
    for i= 1:m
            X(i,:) = (reshape(X(i,:), 28, 28)')(:);
    end
    
    %MNISTデータの60000画像の内、最初の5000はValidation用、5001からはTrain用
    X = X(5001:end,:); 
 
    
    fid = fopen(labelfile, 'r', 'b');
    magic_number = fread(fid, 1, 'int32');
    
    %mは60000になる
    m = fread(fid, 1, 'int32');

    % 60000Byte分をReadする
    yd = fread(fid, [m 1], 'uint8');

    %ydには　数字0から9が格納させている
    %Octaveは1-Baseインデックスなので、0を10に置き換える
    yd(yd==0) = 10;

    %   ydのデータは下記のようになっている
    %   9
    %   7
    %   0
    %   ...

    %今度は one hot vectorに変換する
    %  9の場合[ 0 0 0 0 0 0 0 0 1 0 ]
    %  7の場合[ 0 0 0 0 0 0 1 0 0 0 ]
    %  0の場合[ 0 0 0 0 0 0 0 0 0 1 ] 

    % 行ごとの開始Indexを作成する
    %  0,  10, 20, 30, ... のVectorになる
    index_offset = [(0:1:m-1) * 10]';

    % ydの数字 + Index_offset
    % 1行目の9  　flat_index = 0 + 9 
    % 2行目の7     flat_index =  10 + 7
    % 3行目の0     flat_index = 10 + 10
    % flat_index は  M x 1
    flat_index = yd + index_offset;

    %yを0で初期化する Mx10のMatrixを作成する
    y = zeros(m, 10);

    % M*10 x 1のVectorにreshapeする
    y = reshape(y', m*10, 1);

    y(flat_index) = 1;

    % 10xM のMatrixにReshapeする
    y = reshape(y, 10, m);

    %yを回転させる Mx10になる
    y = y';
   
    %Xと同様5001からTrainデータをRead
    y = y(5001:end,:);

end
