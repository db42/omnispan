Your serving layer (Project 2) stays largely the same. Tenant comes in, API key resolves, rate limit checked, request enters queue. The queue dispatcher now talks to a cluster registry instead of a single local model. The registry says: here are the 45 available model instances, here is their current load, send this request here.

```
Tenant request
     |
API Gateway (identity + enforcement)
     |
Queue + Scheduler (which instance gets this request)
     |
Cluster Registry (where are the instances, what is their load)
     |
Model Instance 1    Model Instance 2    ...    Model Instance 45
(GPU 1+2)           (GPU 3+4)                  (GPU 99+100)
```

Your Project 2 code is everything above the cluster registry. Project 1 is the cluster registry and everything below it.

-----------------

The Java world you know
Client
  → Nginx (reverse proxy, SSL termination)
  → Tomcat (web server + servlet container)
  → Your app (business logic, DB calls)
Tomcat handles HTTP, threading, connection management. Your app handles business logic. They are separate concerns.

The threading model is:
Tomcat process
  └── thread pool (managed by Tomcat)
        └── thread picks up request
              └── calls your app code (same process, same JVM)

Your app code runs inside Tomcat's thread. One process. No IPC.

The inference serving world
Client
  → Nginx (same role, nothing changes)
  → TorchServe / Triton (handles HTTP + manages model workers)
  → Model workers (processes that hold model in memory, do inference)

TorchServe is different:

TorchServe frontend process (Java actually, handles HTTP)
  └── sends request over a socket (IPC)
        └── model worker process 1 (Python, holds model in GPU memory)
        └── model worker process 2
        └── model worker process 3

------------------
