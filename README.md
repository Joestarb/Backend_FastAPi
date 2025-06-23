# Proyecto FastAPI: Predicción y Gestión de Usuarios

Este proyecto es una API desarrollada con **FastAPI** que implementa endpoints para predicción de salud mental basada en el uso de redes sociales y gestión de usuarios. Utiliza una arquitectura modular, facilitando la escalabilidad y el mantenimiento.

## Estructura del Proyecto

```
fastApi/
│   main.py                  # Punto de entrada de la aplicación
│   requirements.txt         # Dependencias del proyecto
│   Social Media.csv         # Dataset para predicción
│
└── app/
    ├── api/
    │   └── v1/
    │       ├── routers/     # Endpoints agrupados por dominio
    │       │   ├── users.py
    │       │   └── predict.py
    │       ├── schemas/     # Esquemas Pydantic
    │       │   └── user.py
    │       └── services/    # Lógica de negocio
    │           └── user_service.py
    ├── core/                # Configuración global y seguridad
    │   ├── config.py
    │   └── security.py
    ├── db/                  # Configuración de la base de datos
    │   ├── base.py
    │   └── session.py
    ├── models/              # Modelos ORM
    │   └── user.py
    └── utils/               # Funciones utilitarias
        └── helpers.py
```

## Principales Endpoints

- **GET /api/v1/users**: Lista de usuarios de ejemplo.
- **GET /api/v1/predict/mental_health?avg_daily_usage=5**: Predice el puntaje de salud mental según las horas promedio de uso diario de redes sociales. Devuelve una imagen con la predicción.

## Instalación y Ejecución

### 1. Crear el entorno virtual

```bash
python -m venv .venv
```

### 2. Activar el entorno virtual (PowerShell en Windows)

```powershell
.\.venv\Scripts\Activate.ps1
```

> Si estás en Git Bash o Linux, usa:
>
> ```bash
> source .venv/bin/activate
> ```

### 3. Instalar FastAPI y Uvicorn

```bash
pip install fastapi uvicorn
```

### 4. (Opcional) Instalar soporte para formularios

> Necesario si vas a subir archivos o usar `Form(...)` en tus endpoints.

```bash
pip install python-multipart
```

### 5. Instalar las dependencias del proyecto

```bash
pip install -r requirements.txt
```

### 6. Ejecutar el servidor de desarrollo

```bash
uvicorn main:app --reload
```

---

## Documentación automática

Una vez iniciado el servidor, puedes acceder a la documentación generada por FastAPI:

- Swagger UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Redoc (si está habilitado): [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Requisitos

- Python 3.10 o superior (preferentemente)
- pip actualizado (`python -m pip install --upgrade pip`)

---

## Notas

- El archivo `Social Media.csv` debe estar en la raíz del proyecto para que el endpoint de predicción funcione correctamente.
- El proyecto está preparado para escalar y añadir nuevos módulos fácilmente.

---

Desarrollado con FastAPI, Pydantic, SQLAlchemy, pandas, scikit-learn y matplotlib.
