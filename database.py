import sqlite3
from typing import List, Dict, Optional
from pathlib import Path
import time

class SmartRetailDB:
    def __init__(self, db_path: str = "smart_retail.db"):
        self.db_path = db_path
        # Delete existing database if it exists
        if Path(db_path).exists():
            Path(db_path).unlink()
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create products table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            description TEXT,
            category TEXT,
            brand TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create carts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS carts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create cart_items table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cart_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cart_id INTEGER,
            product_id TEXT,
            quantity INTEGER DEFAULT 1,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cart_id) REFERENCES carts (id),
            FOREIGN KEY (product_id) REFERENCES products (id)
        )
        ''')

        # Insert default products if they don't exist
        default_products = [
            ('BOTTLE_001', 'Milton Water Bottle', 499.00, '1L Water Bottle', 'Kitchen', 'Milton'),
            ('HEADPHONE_001', 'HP Gaming Headphone', 1599.00, 'HP Gaming Headphone with Noise Cancellation', 'Electronics', 'HP')
        ]
        
        for product in default_products:
            cursor.execute('''
            INSERT OR IGNORE INTO products (id, name, price, description, category, brand)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', product)

        # Create an active cart if none exists
        cursor.execute('''
        INSERT OR IGNORE INTO carts (status) VALUES ('active')
        ''')

        conn.commit()
        conn.close()

    def get_product(self, product_id: str) -> Dict:
        """Get product details by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM products WHERE product_id = ?
        ''', (product_id,))
        
        columns = [description[0] for description in cursor.description]
        product = dict(zip(columns, cursor.fetchone())) if cursor.fetchone() else None
        
        conn.close()
        return product

    def add_to_cart(self, product_id: str):
        """Add a product to the cart."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get active cart
            cursor.execute('SELECT id FROM carts WHERE status = "active" LIMIT 1')
            cart_id = cursor.fetchone()[0]
            
            # Check if product is already in cart
            cursor.execute('''
            SELECT id, quantity FROM cart_items 
            WHERE cart_id = ? AND product_id = ?
            ''', (cart_id, product_id))
            
            item = cursor.fetchone()
            if item:
                # Update quantity if already in cart
                cursor.execute('''
                UPDATE cart_items 
                SET quantity = quantity + 1
                WHERE id = ?
                ''', (item[0],))
            else:
                # Add new item to cart
                cursor.execute('''
                INSERT INTO cart_items (cart_id, product_id, quantity)
                VALUES (?, ?, 1)
                ''', (cart_id, product_id))
            
            conn.commit()
        except Exception as e:
            print(f"Error adding to cart: {e}")
        finally:
            conn.close()

    def remove_from_cart(self, product_id: str):
        """Remove a product from the cart."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get active cart
            cursor.execute('SELECT id FROM carts WHERE status = "active" LIMIT 1')
            cart_id = cursor.fetchone()[0]
            
            # Remove item from cart
            cursor.execute('''
            DELETE FROM cart_items 
            WHERE cart_id = ? AND product_id = ?
            ''', (cart_id, product_id))
            
            conn.commit()
        except Exception as e:
            print(f"Error removing from cart: {e}")
        finally:
            conn.close()

    def get_active_cart(self):
        """Get all items in the active cart."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.id, p.name, p.price, c.quantity
                FROM products p
                JOIN cart_items c ON p.id = c.product_id
                WHERE c.cart_id = (
                    SELECT id FROM carts WHERE status = 'active' LIMIT 1
                )
            ''')
            items = cursor.fetchall()
            return [{'id': item[0], 'name': item[1], 'price': item[2], 'quantity': item[3]} 
                   for item in items]
        except Exception as e:
            print(f"Error getting cart: {e}")
            return []

    def get_cart_history(self, limit: int = 100) -> List[Dict]:
        """Get cart history including removed items."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT p.*, c.added_at, c.removed_at
        FROM products p
        JOIN cart c ON p.product_id = c.product_id
        ORDER BY c.added_at DESC
        LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        items = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return items 