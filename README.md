# Arquitectura MVC con FastAPI

Este proyecto implementa una arquitectura **MVC (Modelo-Vista-Controlador)** usando FastAPI.

## Estructura de Carpetas

```
fastApi/
│   main.py                # Punto de entrada de la aplicación
│
├── models/                # Modelos de datos (Pydantic)
│   └── item.py
│
├── controllers/           # Lógica de negocio y rutas
│   └── item_controller.py
│
└── views/                 # Vistas y manejo de respuestas personalizadas
    └── error_views.py
```

## ¿Cómo funciona la arquitectura MVC en FastAPI?

- **Modelos (`models/`)**: Definen la estructura de los datos usando Pydantic. Por ejemplo, `Item` representa un recurso de la API.
- **Controladores (`controllers/`)**: Contienen la lógica de negocio y definen las rutas de la API. Por ejemplo, `item_controller.py` gestiona las operaciones CRUD sobre los items.
- **Vistas (`views/`)**: Gestionan la presentación de las respuestas y el manejo de errores personalizados. Por ejemplo, `error_views.py` define respuestas para errores 404.
- **main.py**: Es el punto de entrada, donde se inicializa FastAPI, se incluyen los routers de los controladores y se configuran los manejadores de errores y middlewares.

## Ejecución

1. Instala las dependencias:
   ```bash
   pip install fastapi uvicorn
   ```
2. Ejecuta el servidor:
   ```bash
   uvicorn main:app --reload
   ```
3. Accede a la documentación interactiva en [http://localhost:8000/docs](http://localhost:8000/docs)

---

Esta estructura facilita la escalabilidad y el mantenimiento del proyecto, separando claramente la lógica de datos, negocio y presentación.

# Arquitectura para una aplicación grande con FastAPI

Esta estructura está pensada para proyectos grandes y escalables, siguiendo buenas prácticas de organización y separación de responsabilidades.

## Estructura de carpetas

```
/app
    /api
        /v1
            /routers        # Rutas (endpoints) agrupadas por dominio
            /schemas        # Esquemas Pydantic para validación y serialización
            /services       # Lógica de negocio de cada dominio
    /core                   # Configuración y utilidades globales (seguridad, settings, etc)
    /models                 # Modelos ORM (ej. SQLAlchemy)
    /db                     # Configuración de la base de datos
    /utils                  # Funciones utilitarias generales
    main.py                 # Punto de entrada de la app
/tests                      # Pruebas automáticas
```

## ¿Cómo funciona?

- **Routers**: Cada archivo en `routers` define endpoints relacionados (por ejemplo, usuarios, productos, etc.) y se incluyen en la app principal.
- **Schemas**: Definen la estructura de los datos de entrada/salida usando Pydantic.
- **Services**: Contienen la lógica de negocio, separados de los endpoints.
- **Models**: Modelos ORM para la base de datos.
- **Core**: Configuración global, seguridad, variables de entorno, etc.
- **DB**: Configuración de la conexión y sesión de la base de datos.
- **Utils**: Funciones auxiliares reutilizables.
- **Tests**: Pruebas unitarias y de integración.

## Ejemplo de flujo

1. El usuario hace una petición a un endpoint definido en un router.
2. El router valida los datos usando un schema.
3. El router llama a un servicio para ejecutar la lógica de negocio.
4. El servicio interactúa con los modelos y la base de datos si es necesario.
5. Se retorna la respuesta al usuario.

## Ejecución

Instala las dependencias necesarias:

```bash
pip install fastapi uvicorn sqlalchemy pydantic
```

Ejecuta la aplicación:

```bash
uvicorn app.main:app --reload
```

---

Esta arquitectura permite escalar y mantener el proyecto de forma ordenada y profesional.
