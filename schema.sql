DROP TABLE IF EXISTS aaic

CREATE TABLE aaic(
    id INTEGER PRIMARY KEY,
    asin VARCHAR(15),
    brand VARCHAR(100),
    color VARCHAR(20),
    medium_image_url TEXT,
    product_type_name VARCHAR(20),
    title TEXT,
    formatted_price VARCHAR(5));