import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import mysql.connector
import requests
import numpy as np
import os
import openai
from mysql.connector import Error

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

image_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",  
        password="sepai",  # Replace with your MySQL password
        database="foodvision"
    )

def fetch_user_details(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM user_details WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

def update_user_allergens(user_id, allergens):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE user_details SET allergens = %s WHERE user_id = %s", (",".join(allergens), user_id))
    conn.commit()
    conn.close()

def add_new_user(user_id, allergens):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_details (user_id, allergens) VALUES (%s, %s)", (user_id, ",".join(allergens)))
    conn.commit()
    conn.close()


# Function to search specific product sent by Image or Text using Openfoodfacts API
def search_product_by_name(product_name):
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        'search_terms': product_name,
        'action': 'process',
        'json': 1,
        'fields': 'product_name,ingredients_text,ingredients_text_with_allergens',
        'lc': 'en' , 
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('products', [])
    return []


def suggest_alternative(product_name, allergens):
    try:
        # Connect to the database
        connection = mysql.connector.connect(
            host='localhost',    
            user='root',       
            password="sepai",  # Replace with your MySQL password
            database="foodvision"
        )
        
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            
            # Fetch the alternative from the database
            for allergen in allergens:
                query = ("SELECT alternative FROM alternates "
                         "WHERE product_name = %s AND allergen = %s LIMIT 1")
                cursor.execute(query, (product_name, allergen))
                result = cursor.fetchone()
                
                if result:
                    return f"Suggested Alternative: {result['alternative']}"
            
            return "No suitable alternative found in the database."
    
    except Error as e:
        return f"Error connecting to the database: {str(e)}"
    
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()

# Function to search products with Eng Lang details only
def filter_english_products(products):
    filtered_products = []
    for product in products:
        ingredients = product.get("ingredients_text", "").lower()
        # Check if the product's text contains common English words
        if any(word in ingredients for word in ["sugar","salt","oil", "milk", "hazelnut", "cocoa", "wheat", "nuts", "soy", "peanut", "jaggery","egg","flour","seeds"]):
            filtered_products.append(product)
    return filtered_products

def detect_allergens_in_product(ingredients, allergens_to_avoid):
    ingredients = ingredients.lower() if ingredients else ""
    return [allergen for allergen in allergens_to_avoid if allergen.lower() in ingredients]


# Image Processing functions
def extract_features(img_path, model):
        """Extract features from an image using a pre-trained model."""
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features

# GUI application for user registration and login
class FoodAllergenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Allergen Detection System")
        self.root.geometry("500x500")
        self.root.config(bg="#f4f4f9")
        
        self.frames = {}
        self.user_id = None 
        self.user_allergens = []  

        self.frames['login'] = self.create_login_frame()
        self.frames['new_user'] = self.create_new_user_frame()
        self.frames['dashboard'] = self.create_dashboard_frame()

        self.show_frame('login')

    def create_login_frame(self):
        frame = tk.Frame(self.root, bg="#f4f4f9")
        frame.pack(fill="both", expand=True)

        title_label = tk.Label(frame, text="Food Allergen Detection System", font=("Helvetica", 18, "bold"), bg="#f4f4f9")
        title_label.pack(pady=20)

        user_id_label = tk.Label(frame, text="Enter User ID:", font=("Helvetica", 12), bg="#f4f4f9")
        user_id_label.pack(pady=5)

        user_id_entry = tk.Entry(frame, font=("Helvetica", 12), width=20)
        user_id_entry.pack(pady=5)
        
        sign_in_button = tk.Button(frame, text="Sign In", command=lambda: self.sign_in(user_id_entry.get()), font=("Helvetica", 12), bg="#4CAF50", fg="white", width=15)
        sign_in_button.pack(pady=10)

        new_user_button = tk.Button(frame, text="New User", command=self.show_new_user_form, font=("Helvetica", 12), bg="#008CBA", fg="white", width=15)
        new_user_button.pack(pady=10)

        return frame

    def create_new_user_frame(self):
        frame = tk.Frame(self.root, bg="#f4f4f9")
    
        new_user_name_label = tk.Label(frame, text="Enter Username:", font=("Helvetica", 12), bg="#f4f4f9")
        new_user_name_label.grid(row=0, column=0, padx=10, pady=10)

        new_user_name_entry = tk.Entry(frame, font=("Helvetica", 12), width=20)
        new_user_name_entry.grid(row=0, column=1, padx=10, pady=10)

        new_user_allergens_label = tk.Label(frame, text="Enter Allergens (comma-separated):", font=("Helvetica", 12), bg="#f4f4f9")
        new_user_allergens_label.grid(row=1, column=0, padx=10, pady=10)

        new_user_allergens_entry = tk.Entry(frame, font=("Helvetica", 12), width=20)
        new_user_allergens_entry.grid(row=1, column=1, padx=10, pady=10)

        create_new_user_button = tk.Button(frame, text="Create Account", command=lambda: self.create_new_user(new_user_name_entry.get(), new_user_allergens_entry.get()), font=("Helvetica", 12), bg="#4CAF50", fg="white", width=15)
        create_new_user_button.grid(row=2, column=0, columnspan=2, pady=20)

        return frame

    def create_dashboard_frame(self):
        frame = tk.Frame(self.root, bg="#f4f4f9")
        
        self.allergens_label = tk.Label(frame, text="", font=("Helvetica", 12), bg="#f4f4f9")
        self.allergens_label.pack(pady=10)

        self.update_allergens_button = tk.Button(frame, text="Update Allergens", command=self.show_update_allergens_input, font=("Helvetica", 12), bg="#FF9800", fg="white", width=20)
        self.update_allergens_button.pack(pady=10)

        self.product_name_label = tk.Label(frame, text="Enter Product Name:", font=("Helvetica", 12), bg="#f4f4f9")
        self.product_name_label.pack(pady=10)

        self.product_name_entry = tk.Entry(frame, font=("Helvetica", 12), width=25)
        self.product_name_entry.pack(pady=10)

        self.search_button = tk.Button(frame, text="Search", command=self.enter_product_as_text, font=("Helvetica", 12), bg="#4CAF50", fg="white", width=20)
        self.search_button.pack(pady=10)

        self.upload_label = tk.Label(frame, text="Upload Product Image:", font=("Helvetica", 12), bg="#f4f4f9")
        self.upload_label.pack(pady=10)

        self.upload_button = tk.Button(frame, text="Upload Image", command=self.upload_and_recognize_image, font=("Helvetica", 12), bg="#4CAF50", fg="white", width=20)
        self.upload_button.pack(pady=10)

        self.recognized_product_label = tk.Label(frame, text="", font=("Helvetica", 12), bg="#f4f4f9")
        self.recognized_product_label.pack(pady=10)

        return frame

    def sign_in(self, user_id):
        if not user_id:
            messagebox.showwarning("Input Error", "Please enter a User ID")
            return
        user_details = fetch_user_details(user_id)
        if user_details:
            self.user_id = user_id 
            self.show_dashboard(user_details)
        else:
            messagebox.showerror("User Not Found", "No user found with that ID")

    def show_new_user_form(self):
        self.show_frame('new_user')

    def create_new_user(self, username, allergens_str):
        if not username or not allergens_str:
             messagebox.showwarning("Input Error", "Please fill in both username and allergens.")
             return
    
        allergens = [allergen.strip() for allergen in allergens_str.split(",")]

        conn = get_db_connection()
        cursor = conn.cursor()

        query = "INSERT INTO user_details (username, allergens) VALUES (%s, %s)"
        cursor.execute(query, (username, ",".join(allergens)))
        conn.commit()
        conn.close()

        messagebox.showinfo("Success", f"User '{username}' has been successfully created!")
        self.show_login_frame() 

    def show_login_frame(self):
        self.show_frame('login')  

    def show_dashboard(self, user_details):
        self.user_allergens = user_details['allergens'].split(',') 
        self.show_frame('dashboard')
        self.allergens_label.config(text=f"Welcome {user_details['username']}!\nYour Allergens: {user_details['allergens']}")

    def show_frame(self, frame_name):
        for frame in self.frames.values():
            frame.pack_forget()  
        self.frames[frame_name].pack(fill="both", expand=True) 

    def show_update_allergens_input(self):
        self.update_allergens_window = tk.Toplevel(self.root)
        self.update_allergens_window.title("Update Allergens")
        self.update_allergens_window.geometry("400x200")

        label = tk.Label(self.update_allergens_window, text="Enter New Allergens (comma-separated):", font=("Helvetica", 12))
        label.pack(pady=20)

        allergens_entry = tk.Entry(self.update_allergens_window, font=("Helvetica", 12), width=25)
        allergens_entry.pack(pady=10)

        update_button = tk.Button(self.update_allergens_window, text="Update", command=lambda: self.update_allergens(allergens_entry.get()), font=("Helvetica", 12), bg="#4CAF50", fg="white")
        update_button.pack(pady=10)

    def update_allergens(self, allergens_str):
        if not allergens_str:
            messagebox.showwarning("Input Error", "Please enter allergens")
            return
        
        new_allergens = [allergen.strip() for allergen in allergens_str.split(",")]
        update_user_allergens(self.user_id, new_allergens)
        self.user_allergens = new_allergens 
        print(f"Updated allergens: {self.user_allergens}") 
        messagebox.showinfo("Allergy Updated", "Your allergens have been updated successfully!")
        self.update_allergens_window.destroy()  

    #Module to recognise a product name from a uploaded image.   
    def recognize_product_from_image(self, img_path):
        print("Current working directory:", os.getcwd())
        dataset_features = np.load(r'C:\Users\seema\Desktop\Pratyush\Food Vision 2\dataset\extracted_features.npy')
        dataset_image_paths = np.load(r'C:\Users\seema\Desktop\Pratyush\Food Vision 2\dataset\image_paths.npy', allow_pickle=True)

        # Extract features from the query image
        query_features = extract_features(img_path,image_model)
        dataset_features = np.array([f.flatten() for f in dataset_features])
    
        # Find the most similar image
        similarities = cosine_similarity(query_features.reshape(1, -1), dataset_features)
        most_similar_idx = np.argmax(similarities)
        most_similar_image_path = dataset_image_paths[most_similar_idx]
    
        # Extract the product name from the file name
        product_name = os.path.splitext(os.path.basename(most_similar_image_path))[0]
        return product_name    
    
    def upload_and_recognize_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            return
    
        recognized_product_name = self.recognize_product_from_image(file_path)
        if recognized_product_name:
            self.recognized_product_label.config(text=f"Recognized Product: {recognized_product_name}")
            products = search_product_by_name(recognized_product_name)
        else:
            messagebox.showwarning("Error", "Could not recognize the product. Please try another image.")

        if not products:
            messagebox.showinfo("No Results", f"No results found for '{recognized_product_name}'")
            return    

        english_products = filter_english_products(products)
        if not english_products:
            messagebox.showinfo("No Results", f"No Eng results found for '{recognized_product_name}'")
            return
        
        found_allergens = [] 
        for product in english_products[:5]:
            product_name = product.get("product_name", "Unknown")
            if not product_name:
                continue
            print(f"Product: {product_name} ")

            ingredients = product.get("ingredients_text", "")     
            allergens_text = product.get("ingredients_text_with_allergens", "")
            
            combined_text = f"{ingredients} {allergens_text}".strip()

            allergens_in_product = detect_allergens_in_product(combined_text, self.user_allergens) 
            print(f"Debug: ALLERGENS FOUND'{allergens_in_product}'")

            if allergens_in_product:
                result_message = f"⚠️ Your Allergens {', '.join(allergens_in_product)} found in Product: {product_name}"
                messagebox.showwarning("Allergens Found", result_message)

                alternative_suggestion = suggest_alternative(product_name, allergens_in_product)
                messagebox.showinfo("Alternative Suggestion", f"Suggested Alternative: {alternative_suggestion}")
                break
            else:
                messagebox.showinfo("Safe", f"Your Allergen NOT FOUND! The product '{product_name}' is safe for you!")    
                break        
    

    def enter_product_as_text(self):
        product_name = self.product_name_entry.get().strip()
        if not product_name:
            messagebox.showwarning("Input Error", "Please enter a product name.")
            return

        products = search_product_by_name(product_name)
        english_products = filter_english_products(products)
        if not english_products:
            messagebox.showinfo("No Results", f"No results found for '{product_name}'")
            return
        
        for product in english_products[:5]: 
            product_name = product.get("product_name", "Unknown")
            if not product_name:
                continue
            print(f"Product: {product_name} ")

        found_allergens = [] 
        for product in english_products[:5]:
            product_name = product.get("product_name", "Unknown")
            if not product_name:  
                continue

            ingredients = product.get("ingredients_text", "")     
            allergens_text = product.get("ingredients_text_with_allergens", "")
            
            combined_text = f"{ingredients} {allergens_text}".strip()
            print(f"Debug: Combined Text for '{product_name}': {combined_text}")

            allergens_in_product = detect_allergens_in_product(combined_text, self.user_allergens) 
            print(f"Debug: ALLERGENS FOUND : '{allergens_in_product}'")

            if allergens_in_product:
                found_allergens.append((product_name, allergens_in_product))  
            else:
                messagebox.showinfo("Safe", f"The product '{product_name}' is safe for you!")                 
    
        if found_allergens:
            limited_allergens = found_allergens[:5]
            result_message = "\n".join([f"⚠️ Allergen found in {product_name}: {', '.join(allergens)}" for product_name, allergens in limited_allergens])
            messagebox.showwarning("Allergens Found", result_message)


if __name__ == "__main__":
    root = tk.Tk()
    app = FoodAllergenApp(root)
    root.mainloop()
