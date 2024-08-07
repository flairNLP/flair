.. _flair_docs_mainpage:

.. title:: Home

.. raw:: html
   :file: _static/html/landing_page_styles.html

.. raw:: html
   :file: _templates/landing-page-banner.html

.. raw:: html
   :file: _templates/landing-page-illustrations.html

.. raw:: html

   <style>
       .bd-header {
           .bd-header__inner {
               .navbar-header-items__end {
                   .navbar-item:first-of-type {
                       button.search-button {
                           :root .dark-mode & {
                              background-image: url("{{ pathto('_static/magnifying_glass_dark.svg', 1) }}");
                           }
                       }
                   }
               }
           }
       }
   </style>

.. toctree::
   :maxdepth: 3
   :hidden:

   Tutorials <tutorial/index>
   API <api/index>
   Contributing <contributing/index>