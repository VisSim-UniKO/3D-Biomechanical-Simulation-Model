import os
import tkinter as tk
from tkinter import ttk

def populate_tree(tree, node, path):
    """Populate the tree with directory contents."""
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):  # Directory
            dir_node = tree.insert(node, "end", text=item, values=(item_path,), open=False)
            populate_tree(tree, dir_node, item_path)  # Recursive population for directories
        else:  # File
            tree.insert(node, "end", text=item, values=(item_path,))

def on_select(event):
    """Handle item selection in the tree."""
    selected_item = tree.selection()
    if selected_item:
        item_path = tree.item(selected_item[0], "values")[0]
        status_label.config(text=f"Selected: {item_path}")

def create_file_explorer():
    """Create and display the file explorer GUI."""
    global tree, status_label

    # Main window
    root = tk.Tk()
    root.title("File Explorer")
    root.geometry("800x600")

    # Treeview to display the directory structure
    tree = ttk.Treeview(root, columns=("Full Path",), show="tree headings", selectmode="browse")
    tree.heading("#0", text="File/Directory Name", anchor="w")
    tree.heading("Full Path", text="Full Path", anchor="w")
    tree.column("#0", stretch=tk.YES, minwidth=200)
    tree.column("Full Path", stretch=tk.YES, minwidth=400)

    # Populate the treeview starting from cwd
    cwd = os.getcwd()
    root_node = tree.insert("", "end", text=os.path.basename(cwd), values=(cwd,), open=True)
    populate_tree(tree, root_node, cwd)

    # Scrollbar for the treeview
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)

    # Layout
    tree.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    # Status label to display selected item
    status_label = ttk.Label(root, text="Select an item to view its path", anchor="w")
    status_label.grid(row=1, column=0, columnspan=2, sticky="ew")

    # Configure row and column resizing
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Bind selection event
    tree.bind("<<TreeviewSelect>>", on_select)

    root.mainloop()

if __name__ == "__main__":
    create_file_explorer()
