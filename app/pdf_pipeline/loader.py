from pathlib import Path
from fastapi import UploadFile


async def save_upload(file: UploadFile):
	"""
	TODO: Save uploaded PDF file to storage
	1. Check upload directory exists (create if needed)
	2. Read file bytes from upload
	3. Write to destination path
	4. Return path and bytes for processing
	"""
	# TODO: Implementation
	return Path(""), b""
