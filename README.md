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

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta el servidor:
   ```bash
   uvicorn main:app --reload
   ```
3. Accede a la documentación interactiva en [http://localhost:8000/docs](http://localhost:8000/docs)

## Notas

- El archivo `Social Media.csv` debe estar en la raíz del proyecto para que el endpoint de predicción funcione correctamente.
- El proyecto está preparado para escalar y añadir nuevos módulos fácilmente.

---

Desarrollado con FastAPI, Pydantic, SQLAlchemy, pandas, scikit-learn y matplotlib.
