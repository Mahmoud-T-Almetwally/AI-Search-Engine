from django.contrib import admin
from django.urls import path

import search.views as SearchViews

urlpatterns = [
    path('admin/', admin.site.urls),
    path("search/ai/", SearchViews.MultiModalSearchView.as_view(), name="multi_modal_search"),
    path("search/keyword/", SearchViews.KeyWordSearchView.as_view(), name="keyword_search"),
]
