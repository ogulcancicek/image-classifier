import pandas as pd
from img2vec_pytorch import Img2Vec
import image_utils

for image_class in ['bird', 'horse', 'dog', 'cat']:
    paths = image_utils.get_images_from_dir(f'./processed_images/{image_class}')
    images = [image_utils.load_image(path) for path in paths]

    img2vec = Img2Vec()
    embeddings = img2vec.get_vec(images)

    print(embeddings.shape)

    df = pd.DataFrame(embeddings)
    df['filepaths'] = paths
    df.to_csv(f'./embeddings/{image_class}_embeddings.csv', index=False)