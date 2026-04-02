from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),	      
               path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
	       path("ProcessDataset", views.ProcessDataset, name="ProcessDataset"),	      
               path("runExisting", views.runExisting, name="runExisting"),
               path("runPropose", views.runPropose, name="runPropose"),
               path("runExtension", views.runExtension, name="runExtension"),
               path("Graphs", views.Graphs, name="Graphs"),	       
]
