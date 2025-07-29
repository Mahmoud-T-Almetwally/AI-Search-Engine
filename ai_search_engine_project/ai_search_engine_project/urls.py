from django.contrib import admin
from django.urls import path

import search.views as SearchViews

urlpatterns = [
    path('admin/', admin.site.urls),
    path("search/", SearchViews.MultiModalSearchView.as_view(), name="multi_modal_search")
]
